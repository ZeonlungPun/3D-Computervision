import cv2,os,re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
def line_angle(line):
    x1,y1,x2,y2 = line
    return np.arctan2(y2-y1, x2-x1)

def point_line_distance(px, py, line):
    x1,y1,x2,y2 = line
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    return abs(A*px + B*py + C) / (np.hypot(A, B) + 1e-6)

def is_collinear(l1, l2, dist_thresh=8):
    x1,y1,x2,y2 = l1
    return (
        point_line_distance(x1,y1,l2) < dist_thresh and
        point_line_distance(x2,y2,l2) < dist_thresh
    )

def check_duplicate(cur_line,lines_list,angle_thresh, dist_thresh):
    for line in lines_list:
        angle = line_angle(line)
        cur_angle= line_angle(cur_line)
        if abs(angle - cur_angle) < angle_thresh:
            if is_collinear(line, cur_line, dist_thresh):
                return True
    return False
def line_homogeneous(l):
    x1,y1,x2,y2 = l
    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])
    return np.cross(p1, p2)

def intersection_h(l1, l2):
    L1 = line_homogeneous(l1)
    L2 = line_homogeneous(l2)

    P = np.cross(L1, L2)
    if abs(P[2]) < 1e-6:
        return None

    return int(P[0]/P[2]), int(P[1]/P[2])

def calculate_d(full_img_path,save_path):
    img= cv2.imread(full_img_path)
    img_save_name = os.path.join(save_path,full_img_path.split('/')[-1])
    h, w = img.shape[:2]

    # 1. ROI left-bottom part
    roi = img[int(h * 0.25):h, 0:int(w * 0.7)]

    # 2.Gaussian filter remove noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Canny
    edges = cv2.Canny(blur, 15, 20)
    # # 4. HoughLinesP
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=50,
        minLineLength=5,
        maxLineGap=60
    )

    # 5. get the lane
    lines_list = []
    length_list = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if abs(slope) >= 0.5:
                if len(lines_list) != 0:
                    cur_line = (x1, y1, x2, y2)
                    duplicate_sign = check_duplicate(cur_line, lines_list, angle_thresh=np.deg2rad(10), dist_thresh=180)
                    if duplicate_sign:
                        continue
                lines_list.append((x1, y1, x2, y2))
                length_list.append(length)

    length_sort = np.argsort(length_list)
    select_lines = np.array(lines_list)[length_sort[-2::]]
    # get the longest line
    all_y = select_lines[:, [1, 3]]
    yb = np.min(all_y)
    # calculate the intersection point
    vx, vy = intersection_h(select_lines[0], select_lines[1])
    print('vy:',vy)
    # last. draw lines
    out = roi.copy()
    for l in select_lines:
        x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if y1 < y2:
            cv2.line(out, (x1, y1), (int(vx), int(vy)), (255, 255, 255), 2)
        else:
            cv2.line(out, (x2, y2), (int(vx), int(vy)), (255, 255, 255), 2)

    cv2.line(out, (0, yb), (roi.shape[1], yb), (0, 0, 255), 2)
    cv2.putText(out, "yb", (10, yb - 5), 1, 1, 0, 2)
    cv2.line(out, (0, vy), (roi.shape[1], vy), (0, 0, 255), 2)
    cv2.putText(out, "yh", (10, vy - 5), 1, 1, 0, 2)
    cv2.circle(out, (int(vx), int(vy)), radius=5, color=(255, 255, 255))
    cv2.imwrite(img_save_name, out)
    lamda = 1458.19
    H = 8.5
    d0 = lamda * H / (yb - vy)

    return d0

if __name__ == '__main__':
    img_name_list = os.listdir("/home/zonekey/Documents/npmcm2020e/highway/")
    img_name_list =sorted(img_name_list, key=lambda x:int(re.search('\d+',x).group()))
    img_main_path ="/home/zonekey/Documents/npmcm2020e/highway"
    save_main_path="/home/zonekey/Documents/npmcm2020e/result"
    d0_list =[]
    for i,img_name in enumerate(img_name_list):
        full_img_path = os.path.join(img_main_path,img_name)

        d0= calculate_d(full_img_path,save_main_path)
        d0_list.append(d0)
    d0_list = savgol_filter(d0_list, window_length=15, polyorder=2)
    theta=list(range(10,35,5))
    time = np.linspace(1, 100, 100)
    H=8.5
    final_d =[]
    plt.figure(figsize=(10, 6))
    for t in theta:
        d= np.array(d0_list)- np.tan(t / 180 * np.pi) * H
        final_d.append(d)
        plt.plot(time, d, label=f'theta = {t}°')

    plt.xlabel('Time ')  # 橫軸名稱
    plt.ylabel('depth Value')  # Y 軸名稱
    plt.title('depth Value over Time with different Theta')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()  # 顯示圖例
    plt.savefig('depth.png')
