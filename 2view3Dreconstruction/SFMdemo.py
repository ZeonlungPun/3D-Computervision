import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# ====== Load ======
imgs = sorted(glob.glob("/home/zonekey/project/3D-Reconstuction/images/dinos/*.ppm"))[:30]
sift = cv2.SIFT_create(nOctaveLayers=5, contrastThreshold=0.01, edgeThreshold=15)
FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))


def get_matches(des1, des2):
    matches = FLANN.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches


# ====== Step 1: Initialization (Frame 0 and 1) ======
img1 = cv2.imread(imgs[0], 0)
img2 = cv2.imread(imgs[1], 0)
h, w = img1.shape

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matches = get_matches(des1, des2)
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Focal self-cal init
f = np.sqrt(w * h)
K = np.array([  # for dino
        [2360, 0, w / 2],
        [0, 2360, h / 2],
        [0, 0, 1]])

# Estimate Pose
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

# Triangulate initial points
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))
pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3d = (pts4d[:3] / pts4d[3]).T

# --- CRITICAL: Track Management ---
# tracks3D: List of global 3D coordinates
# kp_to_3d: Maps {keypoint_index: 3D_point_index} for the "current" frame
tracks3D = pts3d.tolist()
kp_to_3d = {m.trainIdx: i for i, m in enumerate(matches)}

camera_poses = [(np.eye(3), np.zeros((3, 1))), (R, t)]

# ====== Step 2: Incremental frames ======
for i in range(2, len(imgs)):
    img_next = cv2.imread(imgs[i], 0)
    kp_next, des_next = sift.detectAndCompute(img_next, None)

    # Match current frame (i-1) with next frame (i)
    matches = get_matches(des2, des_next)

    # Separate matches into:
    # 1. Points already in 3D (for PnP)
    # 2. New points (for Triangulation)
    obj_pts, img_pts = [], []
    match_indices = []  # indices in 'matches' list that have 3D correspondences

    for idx, m in enumerate(matches):
        if m.queryIdx in kp_to_3d:
            obj_pts.append(tracks3D[kp_to_3d[m.queryIdx]])
            img_pts.append(kp_next[m.trainIdx].pt)
            match_indices.append(idx)

    if len(obj_pts) < 6:
        print(f"Frame {i}: Not enough points for PnP")
        continue

    # ===== PnP to find camera pose =====
    _, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(obj_pts), np.array(img_pts), K, None)
    R_next, _ = cv2.Rodrigues(rvec)
    t_next = tvec
    camera_poses.append((R_next, t_next))

    # ===== Triangulate NEW points =====
    # Find matches that were NOT used in PnP (new geometry)
    new_kp_prev, new_kp_next = [], []
    new_match_indices = []

    for idx, m in enumerate(matches):
        if m.queryIdx not in kp_to_3d:
            new_kp_prev.append(kp2[m.queryIdx].pt)
            new_kp_next.append(kp_next[m.trainIdx].pt)
            new_match_indices.append(idx)

    if len(new_kp_prev) > 0:
        P_prev = K @ np.hstack(camera_poses[-2])
        P_curr = K @ np.hstack(camera_poses[-1])
        pts4d_new = cv2.triangulatePoints(P_prev, P_curr, np.array(new_kp_prev).T, np.array(new_kp_next).T)
        pts3d_new = (pts4d_new[:3] / pts4d_new[3]).T

        # Update global list and create new mapping
        start_idx = len(tracks3D)
        tracks3D.extend(pts3d_new.tolist())

        # New mapping for the current frame
        new_kp_to_3d = {}
        # 1. Map points carried over via PnP
        for idx in match_indices:
            m = matches[idx]
            new_kp_to_3d[m.trainIdx] = kp_to_3d[m.queryIdx]
        # 2. Map the newly triangulated points
        for j, idx in enumerate(new_match_indices):
            m = matches[idx]
            new_kp_to_3d[m.trainIdx] = start_idx + j

    # Prepare for next iteration
    kp_to_3d = new_kp_to_3d
    kp2, des2 = kp_next, des_next

print(f"[DONE] Reconstructed {len(tracks3D)} points")

# ====== Visualization ======
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

tracks3D = np.array(tracks3D)
ax.scatter(tracks3D[:, 0], tracks3D[:, 1], tracks3D[:, 2], s=1, c='gray', alpha=0.5)

# Plot cameras as red points
for R, t in camera_poses:
    cam_center = -R.T @ t
    ax.scatter(cam_center[0], cam_center[1], cam_center[2], c='r', s=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()