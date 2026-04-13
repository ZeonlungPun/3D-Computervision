"""COLMAP depth-map point cloud fusion in pure Python.

This script reads:
  - sparse/0 or dense/0/sparse/cameras.bin
  - sparse/0 or dense/0/sparse/images.bin
  - dense/0/stereo/depth_maps/*.bin

and produces a fused point cloud with:
  - geometric consistency filtering
  - simple occlusion handling via round-trip reprojection checks
  - estimated depth confidence filtering
  - optional voxel downsampling and outlier removal

Notes
-----
1) COLMAP binary model format:
   - cameras.bin, images.bin are the standard COLMAP model binaries.
   - depth map .bin format is: "width&height&channels&" + float32 data.

2) COLMAP pose convention:
   x_cam = R * x_world + t
   where R is from qvec and t is tvec in images.bin.

3) Depth maps usually do not contain an explicit confidence channel.
   Here we estimate confidence using multi-view geometric consistency and
   optional local depth smoothness.

Dependencies
------------
  pip install numpy

Optional:
  pip install scipy open3d

If open3d is unavailable, the script still writes a PLY file.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import math
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -------------------------
# COLMAP model I/O
# -------------------------

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


@dataclasses.dataclass
class Camera:
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        """Return the intrinsic matrix for standard pinhole-like models."""
        if self.model_name == "SIMPLE_PINHOLE":
            f, cx, cy = self.params[:3]
            fx = fy = f
        elif self.model_name == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
        elif self.model_name in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"}:
            fx, fy, cx, cy = self.params[:4]
        else:
            raise ValueError(f"Unsupported camera model for pinhole K: {self.model_name}")
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K


@dataclasses.dataclass
class Image:
    image_id: int
    qvec: np.ndarray  # [qw, qx, qy, qz]
    tvec: np.ndarray  # [tx, ty, tz]
    camera_id: int
    name: str

    @property
    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    @property
    def C(self) -> np.ndarray:
        """Camera center in world coordinates."""
        R = self.R
        t = self.tvec.reshape(3, 1)
        return (-R.T @ t).reshape(3)


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path: str | Path) -> Dict[int, Camera]:
    path = Path(path)
    cameras: Dict[int, Camera] = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id = read_next_bytes(fid, 8, "Ii")
            width, height = read_next_bytes(fid, 16, "QQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params) if num_params > 0 else []
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model_id=model_id,
                model_name=model_name,
                width=int(width),
                height=int(height),
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def read_images_binary(path: str | Path) -> Dict[int, Image]:
    path = Path(path)
    images: Dict[int, Image] = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
            tvec = np.array(read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            camera_id = read_next_bytes(fid, 4, "I")[0]

            # Read null-terminated string.
            name_bytes = bytearray()
            while True:
                ch = fid.read(1)
                if ch == b"\x00":
                    break
                if ch == b"":
                    raise EOFError("Unexpected EOF while reading image name.")
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")

            # Skip 2D points; not needed for fusion here.
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * (8 + 8 + 8), os.SEEK_CUR)

            images[image_id] = Image(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


# -------------------------
# Geometry
# -------------------------


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion [qw, qx, qy, qz] to rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return R


def world_to_cam(Xw: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ Xw.T + t.reshape(3, 1)).T


def cam_to_world(Xc: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R.T @ (Xc.T - t.reshape(3, 1))).T


def project_points(Xw: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixel coordinates.

    Returns:
        uv: (N, 2)
        z:  (N,) positive depth in camera coordinates
        valid: (N,) points with z > 0
    """
    Xc = world_to_cam(Xw, R, t)
    z = Xc[:, 2]
    valid = z > 1e-8
    x = Xc[:, 0] / np.maximum(z, 1e-8)
    y = Xc[:, 1] / np.maximum(z, 1e-8)
    uv = np.empty((len(Xw), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * x + K[0, 2]
    uv[:, 1] = K[1, 1] * y + K[1, 2]
    return uv, z, valid


def backproject_pixel(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.array([x, y, depth], dtype=np.float64)


# -------------------------
# Depth map I/O
# -------------------------


def read_colmap_array_bin(path: str | Path) -> np.ndarray:
    """Read COLMAP .bin matrix format: width&height&channels& + float32 data.

    Returns array with shape (H, W) for single-channel, or (H, W, C) otherwise.
    """
    path = Path(path)
    with path.open("rb") as fid:
        def read_int_until_ampersand() -> int:
            chars = []
            while True:
                c = fid.read(1)
                if c == b"&":
                    break
                if c == b"":
                    raise EOFError(f"Unexpected EOF while reading header from {path}")
                chars.append(c.decode("utf-8"))
            return int("".join(chars))

        width = read_int_until_ampersand()
        height = read_int_until_ampersand()
        channels = read_int_until_ampersand()
        data = np.fromfile(fid, dtype=np.float32)

    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"{path}: got {data.size} values, expected {expected}")

    arr = data.reshape((width, height, channels), order="F")
    arr = np.transpose(arr, (1, 0, 2))
    if channels == 1:
        arr = arr[:, :, 0]
    return arr


def find_depth_map_path(depth_dir: Path, image_name: str) -> Optional[Path]:
    """Match COLMAP depth map file by exact image name prefix.

    Works for names like:
      - image.jpg.photometric.bin
      - image.jpg.geometric.bin
      - image.jpg.bin
    """
    candidates = [
        depth_dir / f"{image_name}.geometric.bin",
        depth_dir / f"{image_name}.photometric.bin",
        depth_dir / f"{image_name}.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# -------------------------
# Fusion helpers
# -------------------------


def bilinear_sample(img: np.ndarray, u: float, v: float) -> float:
    """Bilinear sample a 2D image at floating coordinates."""
    h, w = img.shape[:2]
    if not (0 <= u < w - 1 and 0 <= v < h - 1):
        return np.nan

    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
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
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )


def local_depth_confidence(depth: np.ndarray, u: int, v: int, window: int = 1) -> float:
    """Estimate a local confidence score from depth smoothness.

    The score is in (0, 1], higher is better. It is only a proxy because COLMAP
    depth maps usually do not contain explicit confidence values.
    """
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
    # Smaller local dispersion -> higher confidence.
    return 1.0 / (1.0 + mad / (med + 1e-6))


@functools.lru_cache(maxsize=32)
def _cached_depth_map(path_str: str) -> np.ndarray:
    return read_colmap_array_bin(path_str)


def load_depth_map(path: Path) -> np.ndarray:
    return _cached_depth_map(str(path))


def compute_camera_neighbors(images: Dict[int, Image], neighbor_count: int) -> Dict[int, List[int]]:
    """Return nearest-neighbor image ids for each image based on camera center distance."""
    ids = sorted(images.keys())
    centers = {iid: images[iid].C for iid in ids}
    neighbors: Dict[int, List[int]] = {}
    for i in ids:
        ci = centers[i]
        dists = []
        for j in ids:
            if j == i:
                continue
            d = float(np.linalg.norm(ci - centers[j]))
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        neighbors[i] = [j for _, j in dists[:neighbor_count]]
    return neighbors


@dataclasses.dataclass
class FusionConfig:
    # Pixel subsampling.
    stride: int = 1

    # Multi-view consistency.
    neighbor_count: int = 10
    min_consistent_views: int = 2
    reproj_threshold_px: float = 1.0
    depth_rel_threshold: float = 0.01

    # Depth filtering.
    min_depth: float = 1e-6
    max_depth: float = 1e6
    min_confidence: float = 0.15
    local_confidence_window: int = 1

    # Post-processing.
    voxel_size: float = 0.0
    statistical_outlier_nb_neighbors: int = 0
    statistical_outlier_std_ratio: float = 2.0

    # Performance.
    max_points_per_image: int = 0  # 0 = no limit


@dataclasses.dataclass
class FusedPoint:
    xyz: np.ndarray
    color: Optional[np.ndarray]
    confidence: float


def geometric_consistency_check(
    Xw: np.ndarray,
    ref_image: Image,
    ref_cam: Camera,
    ref_depth_map: np.ndarray,
    nbr_image: Image,
    nbr_cam: Camera,
    nbr_depth_map: np.ndarray,
    reproj_threshold_px: float,
    depth_rel_threshold: float,
) -> Tuple[bool, float]:
    """Check if a world point is consistent with a neighbor depth map.

    The point is projected into the neighbor view. We then compare the neighbor's
    observed depth at that pixel to the depth of the same 3D point.

    Returns:
        consistent, score
    """
    uv_n, z_n, valid_n = project_points(Xw[None, :], nbr_cam.K, nbr_image.R, nbr_image.tvec)
    if not valid_n[0]:
        return False, 0.0
    u, v = float(uv_n[0, 0]), float(uv_n[0, 1])
    h, w = nbr_depth_map.shape[:2]
    if not (0 <= u < w - 1 and 0 <= v < h - 1):
        return False, 0.0

    d_n = bilinear_sample(nbr_depth_map, u, v)
    if not np.isfinite(d_n) or d_n <= 0:
        return False, 0.0

    # Compare the projected depth of Xw in neighbor view to measured depth.
    rel_depth_err = abs(z_n[0] - d_n) / max(d_n, 1e-6)
    if rel_depth_err > depth_rel_threshold:
        return False, 0.0

    # Round-trip reprojection: neighbor measured depth -> world -> ref view.
    Xc_n = backproject_pixel(u, v, d_n, nbr_cam.K)
    Xw_from_n = cam_to_world(Xc_n[None, :], nbr_image.R, nbr_image.tvec)[0]
    uv_r, z_r, valid_r = project_points(Xw_from_n[None, :], ref_cam.K, ref_image.R, ref_image.tvec)
    if not valid_r[0]:
        return False, 0.0
    du = float(uv_r[0, 0] - uv_n[0, 0])
    dv = float(uv_r[0, 1] - uv_n[0, 1])
    reproj_err = math.sqrt(du * du + dv * dv)

    if reproj_err > reproj_threshold_px:
        return False, 0.0

    # A normalized score for ranking.
    score = math.exp(-reproj_err / max(reproj_threshold_px, 1e-6)) * math.exp(-rel_depth_err / max(depth_rel_threshold, 1e-6))
    return True, float(score)


def fuse_depth_maps(
    sparse_path: str | Path,
    depth_map_dir: str | Path,
    config: FusionConfig,
) -> List[FusedPoint]:
    sparse_path = Path(sparse_path)
    depth_map_dir = Path(depth_map_dir)

    cameras = read_cameras_binary(sparse_path / "cameras.bin")
    images = read_images_binary(sparse_path / "images.bin")

    neighbors = compute_camera_neighbors(images, config.neighbor_count)

    fused: List[FusedPoint] = []
    total_used = 0

    for ref_id in sorted(images.keys()):
        ref_img = images[ref_id]
        ref_cam = cameras[ref_img.camera_id]
        depth_path = find_depth_map_path(depth_map_dir, ref_img.name)
        if depth_path is None:
            print(f"[WARN] No depth map found for image: {ref_img.name}")
            continue

        ref_depth = load_depth_map(depth_path)
        if ref_depth.ndim != 2:
            # Use first channel if needed.
            ref_depth = ref_depth[:, :, 0]

        h, w = ref_depth.shape
        if h != ref_cam.height or w != ref_cam.width:
            print(f"[WARN] Depth size mismatch for {ref_img.name}: depth={ref_depth.shape}, camera=({ref_cam.height}, {ref_cam.width})")

        # Neighbor depth maps are loaded lazily.
        nbr_ids = neighbors.get(ref_id, [])
        if not nbr_ids:
            continue

        # Sampling grid.
        ys = np.arange(0, h, config.stride, dtype=np.int32)
        xs = np.arange(0, w, config.stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        points_this_image = 0

        for u, v in zip(grid_x, grid_y):
            d = float(ref_depth[v, u])
            if not np.isfinite(d) or d <= config.min_depth or d >= config.max_depth:
                continue

            # Depth confidence proxy.
            conf_local = local_depth_confidence(ref_depth, int(u), int(v), window=config.local_confidence_window)
            if conf_local < config.min_confidence:
                continue

            Xc = backproject_pixel(float(u), float(v), d, ref_cam.K)
            Xw = cam_to_world(Xc[None, :], ref_img.R, ref_img.tvec)[0]

            consistent_views = 0
            consistency_score = 0.0

            for nbr_id in nbr_ids:
                nbr_img = images[nbr_id]
                nbr_cam = cameras[nbr_img.camera_id]
                nbr_depth_path = find_depth_map_path(depth_map_dir, nbr_img.name)
                if nbr_depth_path is None:
                    continue
                nbr_depth = load_depth_map(nbr_depth_path)
                if nbr_depth.ndim != 2:
                    nbr_depth = nbr_depth[:, :, 0]

                ok, score = geometric_consistency_check(
                    Xw=Xw,
                    ref_image=ref_img,
                    ref_cam=ref_cam,
                    ref_depth_map=ref_depth,
                    nbr_image=nbr_img,
                    nbr_cam=nbr_cam,
                    nbr_depth_map=nbr_depth,
                    reproj_threshold_px=config.reproj_threshold_px,
                    depth_rel_threshold=config.depth_rel_threshold,
                )
                if ok:
                    consistent_views += 1
                    consistency_score += score

                if consistent_views >= config.min_consistent_views:
                    # Early stop once enough supporting views are found.
                    break

            if consistent_views < config.min_consistent_views:
                continue

            # Final confidence combines local smoothness and multi-view support.
            mv_conf = consistency_score / max(consistent_views, 1)
            final_conf = float(0.5 * conf_local + 0.5 * mv_conf)

            if final_conf < config.min_confidence:
                continue

            fused.append(FusedPoint(xyz=Xw, color=None, confidence=final_conf))
            points_this_image += 1
            total_used += 1

            if config.max_points_per_image > 0 and points_this_image >= config.max_points_per_image:
                break

        print(f"[INFO] Fused {points_this_image} points from {ref_img.name}")

    print(f"[INFO] Total raw fused points: {total_used}")
    return fused


# -------------------------
# Post-processing and export
# -------------------------


def voxel_downsample(points: np.ndarray, colors: Optional[np.ndarray], confidences: np.ndarray, voxel_size: float):
    if voxel_size <= 0:
        return points, colors, confidences

    vox = np.floor(points / voxel_size).astype(np.int64)
    buckets = collections.defaultdict(list)
    for idx, key in enumerate(map(tuple, vox)):
        buckets[key].append(idx)

    out_pts = []
    out_cols = [] if colors is not None else None
    out_conf = []

    for idxs in buckets.values():
        pts = points[idxs]
        conf = confidences[idxs]
        w = conf / max(conf.sum(), 1e-8)
        p = (pts * w[:, None]).sum(axis=0)
        out_pts.append(p)
        out_conf.append(float(conf.mean()))
        if colors is not None:
            cols = colors[idxs]
            c = np.clip((cols * w[:, None]).sum(axis=0), 0, 255)
            out_cols.append(c)

    out_pts = np.asarray(out_pts, dtype=np.float64)
    out_conf = np.asarray(out_conf, dtype=np.float64)
    if colors is not None:
        out_cols = np.asarray(out_cols, dtype=np.uint8)
    return out_pts, out_cols, out_conf


def statistical_outlier_removal(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """Simple statistical outlier removal using nearest-neighbor distances.

    Requires scipy if available; otherwise returns input points unchanged.
    """
    if nb_neighbors <= 0 or len(points) < nb_neighbors + 1:
        return np.ones(len(points), dtype=bool)

    try:
        from scipy.spatial import cKDTree
    except Exception:
        print("[WARN] scipy not available; skipping statistical outlier removal.")
        return np.ones(len(points), dtype=bool)

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)  # includes self at index 0
    mean_dists = dists[:, 1:].mean(axis=1)
    mu = mean_dists.mean()
    sigma = mean_dists.std()
    return mean_dists <= mu + std_ratio * sigma


def write_ply(path: str | Path, points: np.ndarray, colors: Optional[np.ndarray] = None, confidences: Optional[np.ndarray] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_color = colors is not None
    has_conf = confidences is not None

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if has_conf:
            f.write("property float confidence\n")
        f.write("end_header\n")

        for i, p in enumerate(points):
            line = [f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"]
            if has_color:
                c = colors[i]
                line += [str(int(c[0])), str(int(c[1])), str(int(c[2]))]
            if has_conf:
                line += [f"{float(confidences[i]):.6f}"]
            f.write(" ".join(line) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fuse COLMAP depth maps into a point cloud in pure Python.")
    parser.add_argument("--sparse_path", type=str, required=True, help="Path to COLMAP sparse model folder containing cameras.bin and images.bin")
    parser.add_argument("--depth_map_dir", type=str, required=True, help="Path to dense/0/stereo/depth_maps")
    parser.add_argument("--output", type=str, required=True, help="Output PLY file")

    parser.add_argument("--stride", type=int, default=1, help="Pixel stride for sampling depth maps")
    parser.add_argument("--neighbor_count", type=int, default=10, help="Number of nearest neighboring views used for consistency checks")
    parser.add_argument("--min_consistent_views", type=int, default=2, help="Minimum number of consistent neighboring views")
    parser.add_argument("--reproj_threshold_px", type=float, default=1.0, help="Reprojection error threshold in pixels")
    parser.add_argument("--depth_rel_threshold", type=float, default=0.01, help="Relative depth difference threshold")
    parser.add_argument("--min_depth", type=float, default=1e-6, help="Minimum valid depth")
    parser.add_argument("--max_depth", type=float, default=1e6, help="Maximum valid depth")
    parser.add_argument("--min_confidence", type=float, default=0.15, help="Minimum confidence score")
    parser.add_argument("--local_conf_window", type=int, default=1, help="Half-window size for local depth confidence estimation")
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Voxel size for downsampling; 0 disables")
    parser.add_argument("--outlier_nb_neighbors", type=int, default=0, help="Statistical outlier removal neighbor count; 0 disables")
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0, help="Statistical outlier removal std ratio")
    parser.add_argument("--max_points_per_image", type=int, default=0, help="Optional cap per image for debugging; 0 disables")

    args = parser.parse_args()

    cfg = FusionConfig(
        stride=args.stride,
        neighbor_count=args.neighbor_count,
        min_consistent_views=args.min_consistent_views,
        reproj_threshold_px=args.reproj_threshold_px,
        depth_rel_threshold=args.depth_rel_threshold,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_confidence=args.min_confidence,
        local_confidence_window=args.local_conf_window,
        voxel_size=args.voxel_size,
        statistical_outlier_nb_neighbors=args.outlier_nb_neighbors,
        statistical_outlier_std_ratio=args.outlier_std_ratio,
        max_points_per_image=args.max_points_per_image,
    )

    fused = fuse_depth_maps(args.sparse_path, args.depth_map_dir, cfg)
    if not fused:
        raise RuntimeError("No points were fused. Check paths, pose/model consistency, and thresholds.")

    points = np.stack([p.xyz for p in fused], axis=0)
    confidences = np.array([p.confidence for p in fused], dtype=np.float64)
    colors = None

    if cfg.voxel_size > 0:
        points, colors, confidences = voxel_downsample(points, colors, confidences, cfg.voxel_size)

    if cfg.statistical_outlier_nb_neighbors > 0 and len(points) > cfg.statistical_outlier_nb_neighbors + 1:
        mask = statistical_outlier_removal(points, cfg.statistical_outlier_nb_neighbors, cfg.statistical_outlier_std_ratio)
        points = points[mask]
        confidences = confidences[mask]
        if colors is not None:
            colors = colors[mask]

    write_ply(args.output, points, colors=colors, confidences=confidences)
    print(f"[DONE] Wrote {len(points)} points to: {args.output}")


if __name__ == "__main__":
    main()
