import cv2
import numpy as np

# 相機內參（Depth）
FX = 685.194580
FY = 685.029114
CX = 640.183655
CY = 351.862518
#RGB bbox → Depth bbox

#bbox 中位數深度（穩定）
def median_depth(depth, bbox):
    x1, y1, x2, y2 = bbox
    roi = depth[y1:y2, x1:x2]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    if roi.size == 0:
        return None
    return np.median(roi)

def measure_object_size(depth, bbox_rgb):
    """
    depth: (800, 1280) float32, unit = meter
    bbox_rgb: (x1,y1,x2,y2) in RGB image
    """

    # 1. RGB → Depth bbox
    bbox_d = bbox_rgb
    x1, y1, x2, y2 = bbox_d

    # 防止越界
    h, w = depth.shape
    x1 = np.clip(x1, 0, w-1)
    x2 = np.clip(x2, 0, w-1)
    y1 = np.clip(y1, 0, h-1)
    y2 = np.clip(y2, 0, h-1)

    # 2. 深度（中位數）
    Z = median_depth(depth, (x1, y1, x2, y2))
    if Z is None:
        return None

    # 3. 寬度（左右）
    XL = (x1 - CX) * Z / FX
    XR = (x2 - CX) * Z / FX
    width = abs(XR - XL)

    # 4. 高度（上下）
    YT = (y1 - CY) * Z / FY
    YB = (y2 - CY) * Z / FY
    height = abs(YB - YT)

    return {
        "width_m": width,
        "height_m": height,
        "depth_m": Z
    }
def visualize_side_by_side(
    rgb,
    depth,
    bbox_rgb,
    size_info
):
    """
    rgb:   (720,1280,3) uint8
    depth: (800,1280)   float32
    bbox_rgb: (x1,y1,x2,y2)
    size_info: output of measure_object_size()
    """

    # ===============================
    # 1. RGB 視覺化
    # ===============================
    rgb_vis = rgb.copy()
    x1, y1, x2, y2 = map(int, bbox_rgb)

    cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if size_info is not None:
        text = (
            f"W: {size_info['width_m']:.3f} mm  "
            f"H: {size_info['height_m']:.3f} mm  "
            f"Z: {size_info['depth_m']:.3f} mm"
        )
    else:
        text = "Invalid depth"

    cv2.putText(
        rgb_vis, text,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    # ===============================
    # 2. Depth 視覺化
    # ===============================
    depth_vis = depth.copy()
    depth_vis = np.nan_to_num(depth_vis)

    depth_vis = cv2.normalize(
        depth_vis, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Depth bbox
    bbox_d = bbox_rgb
    dx1, dy1, dx2, dy2 = bbox_d
    cv2.rectangle(
        depth_vis,
        (dx1, dy1), (dx2, dy2),
        (255, 255, 255), 2
    )

    # ===============================
    # 3. 對齊高度（Depth 800 → RGB 720）
    # ===============================
    depth_vis = cv2.resize(
        depth_vis,
        (rgb_vis.shape[1], rgb_vis.shape[0])
    )

    # ===============================
    # 4. 拼接顯示（1 行 2 列）
    # ===============================
    combined = np.hstack((depth_vis, rgb_vis))

    cv2.imshow("Depth (Left) | RGB (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return  combined

if __name__ == "__main__":
    # hxw=720x1280
    rgb = cv2.imread("/home/punzeonlung/3d/Color2/s5.png")
    # hxw= 800x1280
    depth = cv2.imread("/home/punzeonlung/3d/Depth2/s5d.png", cv2.IMREAD_UNCHANGED)
    h1, w1 = rgb.shape[:2]
    h2, w2 = depth.shape
    diff = h2 - h1
    padding = int(diff / 2)
    depth = depth[padding:padding + h1, :]

    bbox_rgb = (710, 327,784, 471)
    size = measure_object_size(depth, bbox_rgb)


    combined=visualize_side_by_side(
        rgb,
        depth,
        bbox_rgb,
        size,)
    cv2.imwrite('measure_size5.png',combined)
