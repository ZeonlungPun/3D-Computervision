import cv2,time,math
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
_idx_106_68 = np.array([
    1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,   # 面部轮廓
    43,48,49,51,50,         # 左眉 5
    102,103,104,105,101,    # 右眉 5
    72,73,74,86,77,79,80,83,84, # 鼻子 9
    35,41,42,39,37,36,  # 左眼  6
    89,95,96,93,91,90,  # 右眼  6
    52,64,63,71,67,68,61,58,59,53,56,55,65,66,62,70,69,57,60,54, # 嘴巴 20
])
_3D_face_X = [
    # X
        -73.393523, -72.775014, -70.533638,
        -66.850058, -59.790187, -48.368973,
        -34.121101, -17.875411, 0.098749,
        17.477031, 32.648966, 46.372358,
        57.343480, 64.388482, 68.212038,
        70.486405, 71.375822, -61.119406,
        -51.287588, -37.804800, -24.022754,
        -11.635713, 12.056636, 25.106256,
        38.338588, 51.191007, 60.053851 ,
        0.653940, 0.804809, 0.992204 ,
        1.226783, -14.772472, -7.180239,
        0.555920, 8.272499, 15.214351 ,
        -46.047290, -37.674688, -27.883856,
        -19.648268, -28.272965, -38.082418 ,
        19.265868, 27.894191, 37.437529 ,
        45.170805, 38.196454, 28.764989 ,
        -28.916267, -17.533194, -6.684590,
        0.381001, 8.375443, 18.876618 ,
        28.794412, 19.057574, 8.956375 ,
        0.381549, -7.428895, -18.160634 ,
        -24.377490, -6.897633, 0.340663 ,
        8.444722, 24.474473, 8.449166 ,
        0.205322, -7.198266, ]
_3D_face_Y = [
    # Y
        -29.801432, -10.949766, 7.929818,
        26.074280, 42.564390, 56.481080,
        67.246992, 75.056892, 77.061286,
        74.758448, 66.929021, 56.311389,
        42.419126, 25.455880, 6.990805,
        -11.666193, -30.365191, -49.361602,
        -58.769795, -61.996155, -61.033399,
        -56.686759 , -57.391033, -61.902186,
        -62.777713 , -59.302347, -50.190255,
        -42.193790 , -30.993721, -19.944596,
        -8.414541 , 2.598255, 4.751589,
        6.562900 , 4.661005, 2.643046,
        -37.471411, -42.730510, -42.711517,
        -36.754742, -35.134493, -34.919043,
        -37.032306 ,-43.342445, -43.110822,
        -38.086515 ,-35.532024, -35.484289,
        28.612716 , 22.172187, 19.029051,
        20.721118 , 19.035460, 22.394109,
        28.079924 , 36.298248, 39.634575,
        40.395647 , 39.836405, 36.677899,
        28.677771 , 25.475976, 26.014269,
        25.326198 , 28.323008, 30.596216,
        31.408738 , 30.844876, ]
_3D_face_Z = [
    # Z
        47.667532, 45.909403 , 44.842580,
        43.141114, 38.635298 , 30.750622,
        18.456453, 3.609035 , -0.881698,
        5.181201, 19.176563 , 30.770570,
        37.628629, 40.886309 , 42.281449,
        44.142567, 47.140426 ,14.254422,
        7.268147, 0.442051 , -6.606501,
        -11.967398, -12.051204, -7.315098,
        -1.022953, 5.349435 ,11.615746,
        -13.380835, -21.150853, -29.284036,
        -36.948060, -20.132003, -23.536684,
        -25.944448, -23.695741 , -20.858157,
        7.037989, 3.021217 ,1.353629,
        -0.111088, -0.147273 , 1.476612,
        -0.665746, 0.247660 , 1.696435,
        4.894163, 0.282961 , -1.172675,
        -2.240310, -15.934335, -22.611355,
        -23.748437, -22.721995, -15.610679,
        -3.217393, -14.987997 ,-22.554245,
        -23.591626, -22.406106 ,-15.121907,
        -4.785684, -20.893742 ,-22.220479,
        -21.025520, -5.712776 , -20.671489,
        -21.903670, -20.328022 ,
]



#  get rotation vector and translation vector
def get_pose_estimation(img_size, image_points):
    # 3D model points.
    model_points = np.empty((0,3))
    for X,Y,Z in zip(_3D_face_X,_3D_face_Y,_3D_face_Z):
        face_pt=np.array([X,Y,-Z])
        model_points=np.vstack((model_points,face_pt))


    # Camera internals
    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    #print("Camera Matrix :{}".format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    #print("Rotation Vector:\n {}".format(rotation_vector))
    #print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs
def refine(boxes, max_width, max_height, shift=0.15):
    """Refine the face boxes to suit the face landmark detection's needs.

    Args:
        boxes: [[x1, y1, x2, y2], ...]
        max_width: Value larger than this will be clipped.
        max_height: Value larger than this will be clipped.
        shift (float, optional): How much to shift the face box down. Defaults to 0.1.

    Returns:
       Refined results.
    """
    refined = boxes.copy()
    width = refined[:, 2] - refined[:, 0]
    height = refined[:, 3] - refined[:, 1]

    # Move the boxes in Y direction
    shift = height * shift
    refined[:, 1] += shift
    refined[:, 3] += shift
    center_x = (refined[:, 0] + refined[:, 2]) / 2
    center_y = (refined[:, 1] + refined[:, 3]) / 2

    # Make the boxes squares
    square_sizes = np.maximum(width, height)
    refined[:, 0] = center_x - square_sizes / 2
    refined[:, 1] = center_y - square_sizes / 2
    refined[:, 2] = center_x + square_sizes / 2
    refined[:, 3] = center_y + square_sizes / 2

    # Clip the boxes for safety
    refined[:, 0] = np.clip(refined[:, 0], 0, max_width)
    refined[:, 1] = np.clip(refined[:, 1], 0, max_height)
    refined[:, 2] = np.clip(refined[:, 2], 0, max_width)
    refined[:, 3] = np.clip(refined[:, 3], 0, max_height)

    return refined


def get_face_2d_points(face_image, model_path):

    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    net = ort.InferenceSession(model_path, providers=providers)

    output_name = [i.name for i in net.get_outputs()]
    input_name = [i.name for i in net.get_inputs()]
    #[0,255] --> [-127.5,127.5]
    face_image = (face_image.astype(np.float32) -127.5)/128.0


    # 2. 顏色轉換 (從 BGR 轉換為 RGB)
    # 圖像現在是 float32 且在 [-1, 1] 範圍內
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # 3. 轉置維度：從 (H, W, C) 轉換為 (C, H, W)
    transposed_image = np.transpose(rgb_image, (2, 0, 1))  # (C, H, W)

    # 4. 增加批次維度：從 (C, H, W) 轉換為 (1, C, H, W)
    model_input = np.expand_dims(transposed_image, axis=0)

    # 5. 執行模型推斷
    outputs = net.run(output_name, {input_name[0]: model_input.astype(np.float32)})

    # 6. 後處理輸出
    # 假設輸出是 [1, N*2] (N 個點，每個點 2 個座標)
    face_points_array = outputs[0].reshape((-1, 2))

    return face_points_array

def get_original_interest_points(image,box, model_path):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    scale = (192 * 2.0 / 3.0) / max(w, h, 1e-6)
    M = np.zeros((2, 3))
    M[0, 0] = scale
    M[0, 2] = -center_x * scale + 192 / 2
    M[1, 1] = scale
    M[1, 2] = -center_y * scale + 192 / 2
    face_image = cv2.warpAffine(image, M, (192, 192))
    face_points_array = get_face_2d_points(face_image, model_path)

    # [-1,1]--> [0,2] --> [0,192]
    face_points_array += 1.
    # 轉化爲相對於face_image的座標
    face_points_array *= 192 / 2

    # 轉化爲相對於原圖的座標
    M_inv = cv2.invertAffineTransform(M).astype(np.float32)
    N = face_points_array.shape[0]
    homogeneous_points = np.hstack((face_points_array, np.ones((N, 1)))).astype(np.float32)
    restored_points_2d = homogeneous_points @ M_inv.T


    return restored_points_2d[_idx_106_68,:]



# rotation vector to euler angle
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    #print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    Y = int((pitch / math.pi) * 180)
    X = int((yaw / math.pi) * 180)
    Z = int((roll / math.pi) * 180)

    return 0, Y, X, Z


def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie,image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None

        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        #print(euler_angle_str)
        return 0, pitch, yaw, roll

    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None

def PoseEstimateProcess(img,five_kpts):
    size = img.shape
    valide_judge = np.sum( five_kpts == -1000)>0
    if valide_judge:
        return -1000, -1000, -1000,-1000, -1000, -1000,-1000
    ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, five_kpts)
    if ret != True:
        print('get_pose_estimation failed')
        return -1000,-1000,-1000,-1000, -1000, -1000,-1000
    ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
    return pitch, yaw, roll,rotation_vector, translation_vector, camera_matrix, dist_coeffs

def draw_annotation_box(image, rotation_vector, translation_vector,camera_matrix,coef_matrix, line_width=1):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      coef_matrix)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d[:5]], True, (255,255,255), line_width, cv2.LINE_AA)
    cv2.polylines(image, [point_2d[5::]], True, (0,0,255), line_width, cv2.LINE_AA)
    color2 = (255,0,0)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color2, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color2, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color2, line_width, cv2.LINE_AA)

    return image

def VisualizePose(img, image_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    # ... (步驟 1 & 2 保持不變) ...
    nose_3d_point = np.array([[_3D_face_X[31], _3D_face_Y[31], _3D_face_Z[31]]], dtype=np.float64)
    (nose_end_point2D, jacobian) = cv2.projectPoints(
        nose_3d_point,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )

    # 假設 N=68，索引 31 是鼻子點
    nose_index = 31



    # 確保 p1_obs 是 int tuple
    p1_obs_float = image_points[nose_index].flatten()
    p1_obs = (int(p1_obs_float[0]), int(p1_obs_float[1]))



    # --- 關鍵點 2: 投影點 p2 的安全提取 ---
    p2_proj_flat = nose_end_point2D.ravel()
    p2_proj = (int(p2_proj_flat[0]), int(p2_proj_flat[1]))


    #for (x, y) in image_points.reshape(-1, 2):  # 確保 shape 為 (N, 2)
    #    cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 1, cv2.LINE_AA)  # 使用 img 而非 image，並使用 AA 抗鋸齒


    cv2.circle(img, p2_proj, 3, (0, 255, 0), -1)


    cv2.circle(img, p1_obs, 3, (255, 0, 0), -1)

    cv2.line(img, p1_obs, p2_proj, (0, 0, 255), 2)

    return img




if __name__ == '__main__':
    model=YOLO('yolov8n-face.pt')
    image =cv2.imread('/home/zonekey/project/3d_train_test/val/stand_up/_media_nas_videos_t2_test_student.mp4_468.000-470.000_778_373_92x141/10.jpg')
    model_path='./faceori.onnx'


    results = model(image, conf=0.3, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    five_kpt_list = results.keypoints.data.cpu().numpy()
    #print(image.shape)
    #refined_boxes = refine(boxes,image.shape[1],image.shape[0])
    # five points: left eye,right eye, noise, Left Mouth corner, right Mouth corner

    for box, score, five_kpt in zip(boxes, scores, five_kpt_list):
        img_2d_points = get_original_interest_points(image,box,model_path)
        pitch, yaw, roll,rotation_vector, translation_vector, camera_matrix, dist_coeffs=PoseEstimateProcess(image,img_2d_points)
        img_2d_points = np.array(img_2d_points, dtype=int)
        image=draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        #for (x,y) in _2d_points[31,:] :
        #cv2.circle(image,(int(_2d_points[31,0]),int(_2d_points[31,1])),1,(255,0,0),1,1,1)
        cv2.imwrite('show2.png', image)


