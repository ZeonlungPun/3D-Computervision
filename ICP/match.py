import numpy as np
import os,shutil
def read_file_list(filename):
    data = []
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            timestamp = float(parts[0])
            path = parts[1]
            data.append((timestamp, path))
    return data


rgb_list = read_file_list("./desk/rgb.txt")
depth_list = read_file_list("./desk/depth.txt")
matches = []

for rgb_time, rgb_path in rgb_list:

    best_match = None
    best_diff = 1e9

    for depth_time, depth_path in depth_list:

        diff = abs(rgb_time - depth_time)

        if diff < best_diff:
            best_diff = diff
            best_match = (depth_time, depth_path)

    # 限制最大時間差（很重要）
    if best_diff < 0.02:
        matches.append((rgb_path, best_match[1]))



indices = np.linspace(0, len(matches)-1, 580).astype(int)

selected = [matches[i] for i in indices]

for idx, (rgb_path, depth_path) in enumerate(selected):

    # 統一名稱
    filename = f"{idx:06d}.png"

    rgb_out = os.path.join("./testimg/rgb", filename)
    depth_out = os.path.join("./testimg/depth", filename)

    # 複製
    shutil.copy(os.path.join('./desk',rgb_path), rgb_out)
    shutil.copy(os.path.join('./desk',depth_path), depth_out)

    print(f"Saved pair {idx}: {filename}")