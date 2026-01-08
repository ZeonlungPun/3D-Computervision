import cv2
import dlib
import numpy as np
import math
import torch
import torchvision
from pathlib import Path
import time
from utils.general import non_max_suppression, xyxy2xywh


def get_head_pose(shape, frame):
    # 获取关键点坐标
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # 鼻子尖端
        (shape.part(8).x, shape.part(8).y),  # 下巴
        (shape.part(36).x, shape.part(36).y),  # 左眼左角
        (shape.part(45).x, shape.part(45).y),  # 右眼右角
        (shape.part(48).x, shape.part(48).y),  # 左嘴角
        (shape.part(54).x, shape.part(54).y)  # 右嘴角
    ], dtype="double")

    # 3D模型点，对应于图像中的关键点
    model_points = np.array([
        (0.0, 0.0, 0.0),  # 鼻子尖端
        (0.0, -330.0, -65.0),  # 下巴
        (-225.0, 170.0, -135.0),  # 左眼左角
        (225.0, 170.0, -135.0),  # 右眼右角
        (-150.0, -150.0, -125.0),  # 左嘴角
        (150.0, -150.0, -125.0)  # 右嘴角
    ])

    # camera intrstic matrix
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # assume that no distortion of camera len
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # 计算欧拉角
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = -math.degrees(pitch)
        yaw = -math.degrees(yaw)
        roll = math.degrees(roll)

        return pitch, yaw, roll, p1, p2
    else:
        return None, None, None, None, None


def determine_head_pose_state(pitch, yaw, roll):
    orientation = ""
    if abs(yaw) < 15:
        orientation += "Facing Front"
    elif yaw < -15:
        orientation += "Facing Right"
    else:
        orientation += "Facing Left"
    '''
    if abs(pitch) > 15:
        if pitch < 0:
            orientation += ", Looking Down"
        else:
            orientation += ", Looking Up"
    if abs(roll) > 15:
        if roll < 0:
            orientation += ", Tilted Left"
        else:
            orientation += ", Tilted Right"
    '''
    return orientation


def load_model(model_path):

    model = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)['model'].float().eval()
    return model


def process(model, img_path, output_path):
    # read image
    frame =cv2.imread(img_path)
    original_height, original_width, _ = frame.shape
    # BGR2RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    target_size = (640, 640)
    img = cv2.resize(img, target_size)
    img = torch.from_numpy(img).to(torch.float32) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # face detection ,can skip if there is only one person
    img = img.to(torch.device('cuda'))
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in reversed(det.cpu().detach().numpy()):

                x1, y1, x2, y2 = map(int, xyxy)
                # 640x640 --> original height and width
                x1 = int(x1 * (original_width / 640))
                y1 = int(y1 * (original_height / 640))
                x2 = int(x2 * (original_width / 640))
                y2 = int(y2 * (original_height / 640))

                newrect = dlib.rectangle(x1, y1, x2, y2)
                # face key points
                landmarks = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), newrect)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                # calculate head pose
                pitch, yaw, roll, p1, p2 = get_head_pose(landmarks, frame)

                if pitch is not None and yaw is not None and roll is not None:
                    # 绘制鼻尖到投影点的线
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)


                    # cv2.putText(frame, f"Pitch: {pitch:.2f}", (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # cv2.putText(frame, f"Yaw: {yaw:.2f}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # cv2.putText(frame, f"Roll: {roll:.2f}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    pose_state = determine_head_pose_state(pitch, yaw, roll)
                    cv2.putText(frame, pose_state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                    print(
                        f" Face at ({x}, {y}): Pitch={pitch:.2f}, Yaw={yaw:.2f}, Roll={roll:.2f} -> Pose State: {pose_state}")
    cv2.imwrite(frame,'result.png')






if __name__ == '__main__':

    img_path = 'detect_video.png'
    output_path = 'output_video.mp4'

    model_path = 'yolov5_face_model.pt'
    model = load_model(model_path)

    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    process_video(model, img_path, output_path)