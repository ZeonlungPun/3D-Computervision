import open3d as o3d
import numpy as np
import cv2
import glob
import os

# ==========================================
# 1. TUM Freiburg 1 官方參數設定
# ==========================================
# 相機內參 (Freiburg 1)
fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3
width, height = 640, 480

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

# 畸變係數 (Radial and Tangential Distortion)
dist_coeffs = np.array([0.2624, -0.9513, -0.0054, 0.0026, 1.1633], dtype=np.float32)

# Open3D 內參對象
intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 算法參數
depth_scale = 5000.0  # TUM 標準：5000 代表 1 米
depth_trunc = 3.0  # 截斷 3 米以外的背景雜訊
tsdf_voxel_size = 0.008  # TSDF 解析度 (8mm)，數字越小越精細但也越耗內存

# ==========================================
# 2. 預計算去畸變映射表 (加速處理)
# ==========================================
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1
)


# ==========================================
# 3. 核心輔助函數
# ==========================================

def load_processed_rgbd(color_path, depth_path):
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
    return rgbd


def run_colored_icp(source_rgbd, target_rgbd, init_pose):
    """執行帶有大範圍搜索半徑的 Colored ICP"""
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd, intrinsic_o3d)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd, intrinsic_o3d)

    # 必須計算法向量 (Point-to-Plane 需要)
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 多尺度 ICP：起始半徑設為 10cm (0.1)，容忍手持晃動
    voxel_radii = [0.1, 0.05, 0.02]
    max_iters = [50, 30, 14]
    current_transformation = init_pose

    for radius, max_iter in zip(voxel_radii, max_iters):
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                source_pcd, target_pcd, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter)
            )
            current_transformation = result.transformation
        except RuntimeError:
            # 若此層半徑找不到匹配，則繼續使用上一層的結果
            continue

    return current_transformation


# ==========================================
# 4. 主程序 (重建流水線)
# ==========================================

def main():
    print("正在尋找圖像文件...")
    # 請確保你的終端機當前路徑下有 rgb 和 depth 資料夾
    color_files = sorted(glob.glob("/home/zonekey/project/ICP/testimg/rgb/*.png"))
    depth_files = sorted(glob.glob("/home/zonekey/project/ICP/testimg/depth/*.png"))

    if not color_files or not depth_files:
        print("錯誤：找不到文件！請確認腳本執行路徑與數據集結構。")
        return

    n_frames = min(len(color_files), len(depth_files))
    print(f"共找到 {n_frames} 幀數據。開始重建...")

    # 初始化 TSDF 體積 (消除重影的關鍵)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=tsdf_voxel_size,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 初始化全局位姿
    current_pose = np.eye(4)

    # 處理並集成第一幀
    target_rgbd = load_processed_rgbd(color_files[0], depth_files[0])
    volume.integrate(target_rgbd, intrinsic_o3d, np.linalg.inv(current_pose))

    # 設置步長 (stride) 加速測試。若想獲得最完美效果，請設為 1
    stride = 2

    for i in range(stride, n_frames, stride):
        # 1. 載入並去畸變當前幀
        source_rgbd = load_processed_rgbd(color_files[i], depth_files[i])

        # 2. 計算相對位姿
        T = run_colored_icp(source_rgbd, target_rgbd, np.eye(4))

        # 3. 更新全局位姿
        current_pose = current_pose @ T

        # 4. 集成入 TSDF
        volume.integrate(source_rgbd, intrinsic_o3d, np.linalg.inv(current_pose))

        # 5. 更新下一輪的目標幀
        target_rgbd = source_rgbd

        if i % 10 == 0 or i >= n_frames - stride:
            print(f"進度: {i}/{n_frames}")

    # ==========================================
    # 5. 導出與可視化
    # ==========================================
    print("重建完成！正在提取 3D 模型...")

    # 提取 Mesh 網格 (比點雲更平滑)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # 保存結果
    o3d.io.write_triangle_mesh("desk_undistorted_mesh.ply", mesh)
    print("模型已保存為 desk_undistorted_mesh.ply")

    # 可視化
    o3d.visualization.draw_geometries([mesh], window_name="TUM Desk Reconstruction")


if __name__ == "__main__":
    main()