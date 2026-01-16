import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1. 讀取影像
img1 = cv2.imread('/home/zonekey/project/3D-Reconstuction/images/dinos/viff.007.ppm', 0)
img2 = cv2.imread('/home/zonekey/project/3D-Reconstuction/images/dinos/viff.008.ppm', 0)
if img1 is None or img2 is None: raise FileNotFoundError("Check image paths!")

# 2. SIFT 與基礎矩陣 F
sift = cv2.SIFT_create(
    nfeatures=0,
    nOctaveLayers=5,

)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)
# Lowe's ratio test
good_matches = []
pts1, pts2 = [], []
for m, n in matches:
    if m.distance < 0.7* n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
img1 = cv2.imread('/home/zonekey/project/3D-Reconstuction/images/dinos/viff.007.ppm', 0)
img2 = cv2.imread('/home/zonekey/project/3D-Reconstuction/images/dinos/viff.008.ppm', 0)
height,width =img1.shape[0:2]

K = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])
E, mask = cv2.findEssentialMat(
    pts1, pts2, K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

P0 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P1 = K @ np.hstack((R, t))

pts4d = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
pts3d = (pts4d[:3] / pts4d[3]).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = pts3d[:,0]
Y = pts3d[:,1]
Z = pts3d[:,2]

ax.scatter(X, Y, Z, s=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()