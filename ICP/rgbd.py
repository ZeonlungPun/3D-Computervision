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
search_radii = [0.15, 0.08, 0.04]  # 【新增】ICP 搜索範圍 (15cm, 8cm, 4cm)
max_iters = [50, 30, 14]


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

    # 深度圖【必須】使用最近鄰插值 (INTER_NEAREST)，避免產生虛假的深度值
    depth_undistorted = cv2.remap(depth_raw, map1, map2, cv2.INTER_NEAREST)

    # 3. 轉換為 Open3D 格式
    color_o3d = o3d.geometry.Image(color_undistorted)
    depth_o3d = o3d.geometry.Image(depth_undistorted)

    # 4. 創建 RGBD
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    # 從 RGBD 創建點雲
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic
    )
    return pcd


# --- 3. 多尺度彩色 ICP (Colored ICP) ---
def multi_scale_colored_icp(source, target, voxel_sizes, search_radii, max_iters):
    current_transformation = np.eye(4)

    # 在循環外先做一次初步估計
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    for scale in range(len(voxel_sizes)):
        iter_count = max_iters[scale]
        v_size = voxel_sizes[scale]        # 用於下採樣
        s_radius = search_radii[scale]     # 用於尋找匹配點

        # 下採樣
        source_down = source.voxel_down_sample(v_size)
        target_down = target.voxel_down_sample(v_size)

        # 重新估計法向量
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v_size * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v_size * 2, max_nn=30))

        # 執行 Colored ICP 並捕捉異常
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down,
                s_radius, # <--- 使用更大的獨立搜索半徑
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter_count
                )
            )
            current_transformation = result.transformation
        except RuntimeError as e:
            # 如果因為動作太大而找不到點，印出警告並保留上一層的結果繼續
            print(f"  [警告] 尺度 {scale} (半徑 {s_radius}m) 配准失敗，使用上一層結果。")
            continue

    return current_transformation


# --- 4. 主程序：流水線處理 ---
def main():
    # 獲取文件路徑並排序，確保對齊
    # 假設 RGB 是 .jpg 或 .png，深度是 .pgm
    color_files = sorted(glob.glob("/home/zonekey/project/ICP/testimg/rgb/*.png"))
    depth_files = sorted(glob.glob("/home/zonekey/project/ICP/testimg/depth/*.png"))

    if len(color_files) != len(depth_files):
        print("警告：RGB 與 Depth 文件數量不一致！")

    num_frames = min(len(color_files), len(depth_files))

    poses = [np.eye(4)]  # 存儲位姿
    global_pcd = o3d.geometry.PointCloud()

    # 處理第一幀
    print(f"Initializing with frame 0...")
    target_pcd = load_rgbd_pcd(color_files[0], depth_files[0])
    global_pcd += target_pcd

    # 逐幀配准
    for i in range(1, num_frames):
        print(f"Processing frame {i}/{num_frames - 1}...")

        # 載入當前幀 (Source)
        source_pcd = load_rgbd_pcd(color_files[i], depth_files[i])

        # 配准當前幀與前一幀 (Target)
        T = multi_scale_colored_icp(source_pcd, target_pcd, voxel_sizes, search_radii, max_iters)

        # 累加位姿：這是在全局座標系下的位置
        current_pose = poses[-1] @ T
        poses.append(current_pose)

        # 將當前點雲變換到全局座標系並合併
        # 使用 copy() 避免修改到用於下一輪配准的原始 source_pcd
        pcd_to_add = copy.deepcopy(source_pcd)
        global_pcd += pcd_to_add.transform(current_pose)

        # 定期下採樣全局點雲，防止點數爆炸導致速度變慢
        if i % 5 == 0:
            global_pcd = global_pcd.voxel_down_sample(0.01)

        # 準備下一輪：當前幀變為下一輪的目標
        target_pcd = source_pcd

    # --- 5. 結果輸出 ---
    print("Optimization finished. Final downsampling...")
    global_pcd = global_pcd.voxel_down_sample(0.005)

    # 移除離群雜訊（選配，讓結果更乾淨）
    global_pcd, ind = global_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud("result_colored.ply", global_pcd)
    print("Done! Result saved to result_colored.ply")

    # 可視化
    o3d.visualization.draw_geometries([global_pcd])


if __name__ == "__main__":
    main()