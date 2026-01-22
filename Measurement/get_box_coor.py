import cv2

# 全域變數
points = []
img_show = None

def mouse_callback(event, x, y, flags, param):
    global points, img_show

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))

            # 畫點
            cv2.circle(img_show, (x, y), 5, (0, 0, 255), -1)

            # 顯示座標
            cv2.putText(
                img_show,
                f"({x},{y})",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            cv2.imshow("Select 4 Points", img_show)

def select_four_points(image_path):
    global points, img_show

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("影像讀取失敗")

    img_show = img.copy()
    points = []

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", mouse_callback)

    print("左鍵點擊選擇 4 個點")
    print("按 r 重置，按 q 或 ESC 結束")

    while True:
        cv2.imshow("Select 4 Points", img_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            img_show = img.copy()
            points = []
            print("已重置")

        elif key == ord('q') or key == 27:
            break

        if len(points) == 4:
            print("已選擇 4 個點")
            break

    cv2.destroyAllWindows()
    return points

img_path = "/home/punzeonlung/3d/Color2/s5.png"
points = select_four_points(img_path)
print(points)
#[(560, 241), (631, 240), (563, 340), (625, 336)]
#[(647, 274), (790, 272), (646, 343), (791, 339)]
#[(453, 356), (838, 351), (462, 551), (845, 544)]

#[(561, 230), (630, 234), (564, 337), (625, 338)]
#[(556, 484), (619, 483), (559, 579), (613, 584)]

#[(527, 333), (709, 334), (714, 467), (529, 469)]