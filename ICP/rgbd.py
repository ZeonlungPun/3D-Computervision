import open3d as o3d
import numpy as np
import glob
import os
import copy,cv2

# --- 1. 相機參數設定 ---
fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3
width, height = 640, 480
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

# 畸變係數 (Radial and Tangential Distortion)
dist_coeffs = np.array([0.2624, -0.9513, -0.0054, 0.0026, 1.1633], dtype=np.float32)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1
)
# Open3D 內參對象
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


depth_scale = 5000.0
depth_trunc = 3.0
voxel_sizes = [0.05, 0.02, 0.01]   # 點雲解析度
search_radii = [0.15, 0.08, 0.04]  # ICP 搜索範圍 (15cm, 8cm, 4cm)
max_iters = [50, 30, 14]
def estimate_normals_adaptive(pcd, voxel_size):
    radius_normal = voxel_size * 2.5  # 自適應半徑
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

def global_registration_ransac(source, target, voxel_size):
    # 1. 下採樣
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # 2. 法向量估計
    estimate_normals_adaptive(source_down, voxel_size)
    estimate_normals_adaptive(target_down, voxel_size)

    # 3. 計算 FPFH 特徵
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )

    # 4. RANSAC 全局配準
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size*1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
    )
    return result.transformation

# --- 2. 加載 RGB-D 數據並轉為點雲 ---
def load_rgbd_pcd(color_path, depth_path):
    """讀取圖像並使用 OpenCV 進行畸變校正，然後轉為 Open3D RGBD 對象"""
    # 1. 讀取原始圖像
    color_bgr = cv2.imread(color_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if color_bgr is None or depth_raw is None:
        raise FileNotFoundError(f"無法讀取圖像: {color_path} 或 {depth_path}")

    # 2. 去除畸變 (Undistortion)
    # 彩色圖使用雙線性插值 (INTER_LINEAR)，並轉回 RGB
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_undistorted = cv2.remap(color_rgb, map1, map2, cv2.INTER_LINEAR)

    # 深度圖 median filter 去除跳躍值
    depth_raw = cv2.medianBlur(depth_raw, 5)
    #雙邊濾波平滑深度圖
    depth_raw = cv2.bilateralFilter(depth_raw, d=5, sigmaColor=75, sigmaSpace=75)

    # 深度圖使用最近鄰插值 ，避免產生虛假的深度值
    depth_undistorted = cv2.remap(depth_raw, map1, map2, cv2.INTER_NEAREST)

    # 3. 轉換為 Open3D 格式
    color_o3d = o3d.geometry.Image(color_undistorted)
    depth_o3d = o3d.geometry.Image(depth_undistorted)

    # 4. 創建 RGBD
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=True
    )

    # 從 RGBD 創建點雲
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic
    )
    return pcd


# --- 3. 多尺度彩色 ICP (Colored ICP) ---
def multi_scale_colored_icp(source, target, init_transformation, voxel_sizes, search_radii, max_iters):
    current_transformation = init_transformation

    for scale in range(len(voxel_sizes)):
        v_size = voxel_sizes[scale]
        s_radius = search_radii[scale]
        iter_count = max_iters[scale]

        source_down = source.voxel_down_sample(v_size)
        target_down = target.voxel_down_sample(v_size)
        estimate_normals_adaptive(source_down, v_size)
        estimate_normals_adaptive(target_down, v_size)

        result = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down,
            s_radius,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter_count
            )
        )
        current_transformation = result.transformation
    return current_transformation


# --- 4. 主程序：流水線處理 ---
def main():
    # --- 1. 獲取文件路徑 ---
    color_files = sorted(glob.glob("./testimg/rgb/*.png"))
    depth_files = sorted(glob.glob("./testimg/depth/*.png"))

    if len(color_files) != len(depth_files):
        print("警告：RGB 與 Depth 文件數量不一致！")

    num_frames = min(len(color_files), len(depth_files))

    # --- 2. 初始化 ---
    poses = [np.eye(4)]
    global_pcd = o3d.geometry.PointCloud()

    # 處理第一幀
    print("Initializing with frame 0...")
    target_pcd = load_rgbd_pcd(color_files[0], depth_files[0])
    global_pcd += target_pcd

    # --- 3. 逐幀配準 ---
    for i in range(1, num_frames):
        print(f"Processing frame {i}/{num_frames - 1}...")

        # 3.1 載入當前幀
        source_pcd = load_rgbd_pcd(color_files[i], depth_files[i])

        # 3.2 全局粗對齊（FPFH + RANSAC）
        T_init = global_registration_ransac(source_pcd, target_pcd, voxel_sizes[0])

        # 3.3 多尺度 Colored ICP 精對齊
        T_icp = multi_scale_colored_icp(source_pcd, target_pcd, T_init, voxel_sizes, search_radii, max_iters)

        # 3.4 累積位姿到全局
        current_pose = poses[-1] @ T_icp
        poses.append(current_pose)

        # 3.5 將當前幀點雲變換到全局並合併
        pcd_to_add = copy.deepcopy(source_pcd)
        pcd_to_add.transform(current_pose)
        global_pcd += pcd_to_add

        # 3.6 定期下採樣全局點雲
        if i % 5 == 0:
            global_pcd = global_pcd.voxel_down_sample(0.01)

        # 3.7 更新 target_pcd
        target_pcd = source_pcd

    # --- 4. 最終處理 ---
    print("Optimization finished. Final downsampling...")
    global_pcd = global_pcd.voxel_down_sample(0.005)

    # 移除離群點
    global_pcd, ind = global_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    # --- 5. 輸出結果 ---
    o3d.io.write_point_cloud("result_colored.ply", global_pcd)
    print("Done! Result saved to result_colored.ply")

    # 可視化
    o3d.visualization.draw_geometries([global_pcd])

if __name__ == "__main__":
    main()