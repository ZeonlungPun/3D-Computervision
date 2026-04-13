import os
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


# ----------------------------
# COLMAP binary I/O
# ----------------------------
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("EQUIRECTANGULAR", 0),
    12: ("SIMPLE_EQUIRECTANGULAR", 0),
}

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

@dataclass
class Camera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self):
        if self.model_name == "SIMPLE_PINHOLE":
            f, cx, cy = self.params[:3]
            fx = fy = f
        elif self.model_name == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
        else:
            # 你現在做 depth fusion，通常 COLMAP 深度圖對應的是 pinhole 類模型
            fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray   # [qw, qx, qy, qz]
    tvec: np.ndarray   # [tx, ty, tz]
    camera_id: int
    name: str
    bgr: np.ndarray   # [h,w,3]


    @property
    def R(self):
        return qvec2rotmat(self.qvec)

    @property
    def C(self):
        # camera center in world coordinates
        return (-self.R.T @ self.tvec.reshape(3, 1)).reshape(3)

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),         2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id = read_next_bytes(fid, 8, "Ii")
            width, height = read_next_bytes(fid, 16, "QQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params) if num_params > 0 else []
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model_name=model_name,
                width=int(width),
                height=int(height),
                params=np.array(params, dtype=np.float64),
            )
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
            tvec = np.array(read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            camera_id = read_next_bytes(fid, 4, "I")[0]

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * (8 + 8 + 8), os.SEEK_CUR)
            
            img_path = str(path).split('sparse')[0]+'images/'+name
            bgr= cv2.imread(img_path)
            
            images[image_id] = ImagePose(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                bgr=bgr
            )
    return images

# ----------------------------
# Depth map I/O
# ----------------------------
def read_colmap_depth_bin(path):
    with open(path, "rb") as f:
        def read_int():
            chars = []
            while True:
                c = f.read(1)
                if c == b"&":
                    break
                chars.append(c.decode("utf-8"))
            return int("".join(chars))

        width = read_int()
        height = read_int()
        channels = read_int()
        data = np.fromfile(f, dtype=np.float32)

    arr = data.reshape((width, height, channels), order="F")
    arr = np.transpose(arr, (1, 0, 2))  # (H, W, C)
    if channels == 1:
        arr = arr[:, :, 0]
    return arr

def load_depth_maps(depth_dir, images):
    depth_maps = {}
    for img_id, img in images.items():
        p1 = Path(depth_dir) / f"{img.name}.geometric.bin"
        p2 = Path(depth_dir) / f"{img.name}.photometric.bin"
        p3 = Path(depth_dir) / f"{img.name}.bin"
        if p1.exists():
            depth_maps[img_id] = read_colmap_depth_bin(p1)
        elif p2.exists():
            depth_maps[img_id] = read_colmap_depth_bin(p2)
        elif p3.exists():
            depth_maps[img_id] = read_colmap_depth_bin(p3)
    return depth_maps

# ----------------------------
# Geometry
# ----------------------------
def world_to_cam(Xw, R, t):
    return R @ Xw + t

def cam_to_world(Xc, R, t):
    return R.T @ (Xc - t)

def project_point(Xw, cam: Camera, pose: ImagePose):
    Xc = world_to_cam(Xw, pose.R, pose.tvec)
    z = Xc[2]
    if z <= 1e-8:
        return None
    u = cam.K[0, 0] * (Xc[0] / z) + cam.K[0, 2]
    v = cam.K[1, 1] * (Xc[1] / z) + cam.K[1, 2]
    return np.array([u, v, z], dtype=np.float64)

def backproject_pixel(u, v, depth, cam: Camera):
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.array([x, y, depth], dtype=np.float64)

def estimate_normal_from_depth(depth, u, v, cam: Camera):
    """
    用鄰域深度估計法向量，替代 C++ 裡的 normal_map
    """
    h, w = depth.shape
    if u <= 0 or v <= 0 or u >= w - 1 or v >= h - 1:
        return None

    d = depth[v, u]
    dr = depth[v, u + 1]
    dd = depth[v + 1, u]
    if d <= 0 or dr <= 0 or dd <= 0:
        return None

    p = backproject_pixel(u, v, d, cam)
    pr = backproject_pixel(u + 1, v, dr, cam)
    pd = backproject_pixel(u, v + 1, dd, cam)

    n = np.cross(pr - p, pd - p)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None
    return n / norm

def bilinear_sample(img, u, v):
    h, w = img.shape[:2]
    if not (0 <= u < w - 1 and 0 <= v < h - 1):
        return np.nan
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    x1 = x0 + 1
    y1 = y0 + 1
    dx = u - x0
    dy = v - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    if not (np.isfinite(v00) and np.isfinite(v10) and np.isfinite(v01) and np.isfinite(v11)):
        return np.nan

    return (
        v00 * (1 - dx) * (1 - dy) +
        v10 * dx * (1 - dy) +
        v01 * (1 - dx) * dy +
        v11 * dx * dy
    )

def local_depth_confidence(depth, u, v, window=1):
    h, w = depth.shape
    x0 = max(0, u - window)
    x1 = min(w, u + window + 1)
    y0 = max(0, v - window)
    y1 = min(h, v + window + 1)
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    if valid.sum() < 3:
        return 0.0
    vals = patch[valid]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return 1.0 / (1.0 + mad / (med + 1e-6))

def build_overlap_neighbors(images, k=10):
    ids = sorted(images.keys())
    centers = {i: images[i].C for i in ids}
    neighbors = {}
    for i in ids:
        ci = centers[i]
        dists = []
        for j in ids:
            if i == j:
                continue
            d = np.linalg.norm(ci - centers[j])
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        neighbors[i] = [j for _, j in dists[:k]]
    return neighbors

# ----------------------------
# Fusion core
# ----------------------------
def fuse_seed(
    thread_id,
    seed_image_idx,
    seed_row,
    seed_col,
    cameras,
    images,
    depth_maps,
    overlap_neighbors,
    fused_pixel_masks,
    max_depth_error=0.01,
    max_squared_reproj_error=1.0,
    min_cos_normal_error=0.7,
    min_num_pixels=3,
    max_num_pixels=50,
    max_traversal_depth=3,
    min_confidence=0.15,
    bbox_min=None,
    bbox_max=None,
):
 #如果該列表中還有數據，一直遞歸
    fusion_queue = [(seed_image_idx, seed_row, seed_col, 0)]
# reference point
    fused_ref_point = None
    fused_ref_normal = None

    fused_x, fused_y, fused_z = [], [], []
    fused_nx, fused_ny, fused_nz = [], [], []
    fused_r,fused_g,fused_b = [], [], []
    fused_visibility = set()

    while fusion_queue:
        image_idx, row, col, traversal_depth = fusion_queue.pop()
    #檢查該點是否被處理
        if fused_pixel_masks[image_idx][row, col] > 0:
            continue
    #得到深度值
        depth = depth_maps[image_idx][row, col]
        if not np.isfinite(depth) or depth <= 0.0:
            continue
    #內外參數
        cam = cameras[images[image_idx].camera_id]
        pose = images[image_idx]
    #RGB圖像
        bgr  = images[image_idx].bgr[row, col]

        normal = estimate_normal_from_depth(depth_maps[image_idx], col, row, cam)
        if normal is None:
            continue

        # traversal_depth > 0 時，做幾何一致性檢查
        if traversal_depth > 0:
        # reference point 投影到source image
            proj = project_point(fused_ref_point, cam, pose)
            if proj is None:
                continue
        #深度一致性檢查
            depth_error = abs((proj[2] - depth) / depth)
            if depth_error > max_depth_error:
                continue
        #幾何一致性
            col_diff = proj[0] - col
            row_diff = proj[1] - row
            squared_reproj_error = col_diff * col_diff + row_diff * row_diff
            if squared_reproj_error > max_squared_reproj_error:
                continue
        #normal一致性
            cos_normal_error = float(np.dot(fused_ref_normal, normal))
            if cos_normal_error < min_cos_normal_error:
                continue

        # current pixel -> 3D point
        Xc = backproject_pixel(col, row, depth, cam)
        Xw = cam_to_world(Xc, pose.R, pose.tvec)

        # 標記已訪問
        fused_pixel_masks[image_idx][row, col] = 1

        # bounding box filter（如果你有設定）
        if bbox_min is not None and bbox_max is not None:
            if np.any(Xw < bbox_min) or np.any(Xw > bbox_max):
                continue

        fused_x.append(Xw[0])
        fused_y.append(Xw[1])
        fused_z.append(Xw[2])
        fused_nx.append(normal[0])
        fused_ny.append(normal[1])
        fused_nz.append(normal[2])
        fused_b.append(bgr[0])
        fused_g.append(bgr[1])
        fused_r.append(bgr[2])
        fused_visibility.add(image_idx)

        # 第一個像素作為 reference
        if traversal_depth == 0:
            fused_ref_point = Xw.copy()
            fused_ref_normal = normal.copy()

        if len(fused_x) >= max_num_pixels:
            break

        if traversal_depth >= max_traversal_depth - 1:
            continue

        # 擴展到鄰近 view，類似 overlapping_images_
        for next_image_idx in overlap_neighbors[image_idx]:
            next_cam = cameras[images[next_image_idx].camera_id]
            next_pose = images[next_image_idx]

            proj_next = project_point(Xw, next_cam, next_pose)
            if proj_next is None:
                continue

            next_col = int(round(proj_next[0]))
            next_row = int(round(proj_next[1]))

            h, w = depth_maps[next_image_idx].shape[:2]
            if next_col < 0 or next_row < 0 or next_col >= w or next_row >= h:
                continue

            fusion_queue.append((next_image_idx, next_row, next_col, traversal_depth + 1))

    if len(fused_x) < min_num_pixels:
        return None

    # 中位數融合，對應 C++ 的 Median(...)
    px = float(np.median(fused_x))
    py = float(np.median(fused_y))
    pz = float(np.median(fused_z))

    n = np.array([
        np.median(fused_nx),
        np.median(fused_ny),
        np.median(fused_nz),
    ], dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-8:
        return None
    n /= nn
    
    pr = int(np.median(fused_r))
    pg = int(np.median(fused_g))
    pb = int(np.median(fused_b))
    
    return {
        "xyz": np.array([px, py, pz], dtype=np.float64),
        "normal": n,
        "visibility": sorted(list(fused_visibility)),
        "num_pixels": len(fused_x),
        "colour":np.array([pr,pg,pb],dtype=int)
    }

def fuse_all_points(
    sparse_path,
    depth_map_dir,
    stride=2,
    neighbor_k=10,
    min_num_pixels=3,
    max_num_pixels=50,
    max_traversal_depth=3,
    max_depth_error=0.01,
    max_squared_reproj_error=1.0,
    min_cos_normal_error=0.7,
    min_confidence=0.15,
):
    cameras = read_cameras_binary(Path(sparse_path) / "cameras.bin")
    images = read_images_binary(Path(sparse_path) / "images.bin")
    depth_maps = load_depth_maps(depth_map_dir, images)

    overlap_neighbors = build_overlap_neighbors(images, k=neighbor_k)

    # 每張圖一個 fused mask
    fused_pixel_masks = {
        img_id: np.zeros(depth_maps[img_id].shape[:2], dtype=np.uint8)
        for img_id in depth_maps.keys()
    }

    fused_points = []

    for image_idx in sorted(depth_maps.keys()):
        depth = depth_maps[image_idx]
        h, w = depth.shape[:2]

        for row in range(0, h, stride):
            for col in range(0, w, stride):
                if fused_pixel_masks[image_idx][row, col] > 0:
                    continue

                # seed confidence
                conf = local_depth_confidence(depth, col, row, window=1)
                if conf < min_confidence:
                    continue

                result = fuse_seed(
                    thread_id=0,
                    seed_image_idx=image_idx,
                    seed_row=row,
                    seed_col=col,
                    cameras=cameras,
                    images=images,
                    depth_maps=depth_maps,
                    overlap_neighbors=overlap_neighbors,
                    fused_pixel_masks=fused_pixel_masks,
                    max_depth_error=max_depth_error,
                    max_squared_reproj_error=max_squared_reproj_error,
                    min_cos_normal_error=min_cos_normal_error,
                    min_num_pixels=min_num_pixels,
                    max_num_pixels=max_num_pixels,
                    max_traversal_depth=max_traversal_depth,
                )

                if result is None:
                    continue

                fused_points.append(result)

    return fused_points

def write_ply(path, points):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p in points:
            x, y, z = p["xyz"]
            nx, ny, nz = p["normal"]
            r,g,b = p["colour"]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {r} {g} {b}\n")

# Example usage:
points = fuse_all_points(
    sparse_path="/home/zonekey/project/colmap/auto_reconstruction/dense/0/sparse",
    depth_map_dir="/home/zonekey/project/colmap/auto_reconstruction/dense/0/stereo/depth_maps",
    stride=2,
    neighbor_k=10,
    min_num_pixels=3,
    max_traversal_depth=3,
)
write_ply("fused3.ply", points)


