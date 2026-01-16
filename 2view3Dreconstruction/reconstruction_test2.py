import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. SIFT + Fundamental Matrix
# ==============================
def sift_match(img1, img2):
    sift = cv2.SIFT_create(nOctaveLayers=5,    nfeatures=0)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return pts1, pts2, F

# ==============================
# 2. 從F取得projective P,P'
# ==============================
def fundamental_to_camera(F):
    P0 = np.hstack((np.eye(3), np.zeros((3,1))))

    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]; e = e / e[2]

    ex = np.array([[0, -e[2], e[1]],
                   [e[2], 0, -e[0]],
                   [-e[1], e[0], 0]])

    P1 = np.hstack((ex @ F, e.reshape(3,1)))
    return P0, P1

# ==============================
# 3. triangulation → projective 3D
# ==============================
def triangulate(P0, P1, pts1, pts2):
    pts4D = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
    pts4D /= pts4D[3]
    return pts4D[:3].T  # Nx3

# ==============================
# 4. LSD + cluster lines + vanishing pts
# ==============================
def line_detect_clusters(img, k=3):
    lsd = cv2.createLineSegmentDetector(0)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines, _, _, _ = lsd.detect(img)

    dirs=[]; segments=[]
    for l in lines:
        x1,y1,x2,y2 = l[0]
        ang=np.arctan2(y2-y1, x2-x1)
        dirs.append([np.cos(ang), np.sin(ang)])
        segments.append((np.array([x1,y1,1]), np.array([x2,y2,1])))

    # cluster direction vectors
    dirs=np.float32(dirs)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)
    _, labels, _=cv2.kmeans(dirs, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    groups=[[] for _ in range(k)]
    for seg,c in zip(segments, labels.ravel()):
        groups[c].append(seg)

    return groups

def compute_vanish(groups):
    vpts=[]
    for g in groups:
        vs=[]
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                l1=np.cross(g[i][0], g[i][1])
                l2=np.cross(g[j][0], g[j][1])
                v=np.cross(l1,l2)
                if abs(v[2])<1e-6: continue
                vs.append(v/v[2])
        if len(vs):
            vpts.append(np.median(np.array(vs), axis=0))
    return vpts

# ==============================
# 5. Affine upgrade
# ==============================
def affine_upgrade(pts3D_proj, v1, v2):
    l_inf = np.cross(v1, v2)
    π_inf = np.append(l_inf, 0.0)  # make it 4D
    l_inf /= l_inf[2]
    H = np.eye(4)
    H[3, :] = π_inf  # place plane at infinity
    #Hinv=np.linalg.inv(H)

    pts3D_aff=[]
    for X in pts3D_proj:
        Xh=np.array([X[0],X[1],X[2],1.0])
        Xa=H@Xh
        Xa/=Xa[3]
        pts3D_aff.append(Xa)
    return np.array(pts3D_aff)


# ==============================
# ======= Main Pipeline ========
# ==============================

img1 = cv2.imread('/home/zonekey/project/3D-Reconstuction/images/dinos/viff.007.ppm', )
img2 = cv2.imread('/home/zonekey/project/3D-Reconstuction/images/dinos/viff.008.ppm', )
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 1. SIFT + F
pts1, pts2, F = sift_match(gray1, cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

# 2. P,P'
P0, P1 = fundamental_to_camera(F)

# 3. Projective Triangulation
pts3D_proj = triangulate(P0, P1, pts1, pts2)

# 4. LSD → clusters → vanishing pts
groups = line_detect_clusters(img1, k=3)
vpts = compute_vanish(groups)

# 需要至少2個vanishing pts
v1, v2 = vpts[0], vpts[1]

# 5. affine upgrade
pts3D_aff = affine_upgrade(pts3D_proj, v1, v2)

# 6. visualize
X,Y,Z = pts3D_aff[:,0], pts3D_aff[:,1], pts3D_aff[:,2]
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, s=4)
plt.show()
