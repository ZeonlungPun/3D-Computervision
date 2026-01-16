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

# --- 可視化 SIFT 匹配結果 ---
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("SIFT Feature Matching")
plt.axis('off')
plt.savefig("sift_matches.png")
print("SIFT 匹配圖已保存至 sift_matches.png")

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Camera 1
P0 = np.hstack((np.eye(3), np.zeros((3,1))))

# Camera 2 from F
# get epipole e s.t. F e = 0
U, S, Vt = np.linalg.svd(F)
e = Vt[-1]
e = e/e[2]

# Skew of e
ex = np.array([
    [0, -e[2], e[1]],
    [e[2], 0, -e[0]],
    [-e[1], e[0], 0]
])

P1 = np.hstack((ex @ F, e.reshape(3,1)))

# Triangulate
pts4D = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
pts3D = pts4D / pts4D[3]

pts3D = pts3D[:3].T   # N x 3

print(pts3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = pts3D[:,0]
Y = pts3D[:,1]
Z = pts3D[:,2]

ax.scatter(X, Y, Z, s=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()






