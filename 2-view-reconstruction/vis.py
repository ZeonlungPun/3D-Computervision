import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 讀取 CSV
data = np.loadtxt('tripoints3d.csv', delimiter=',', skiprows=1)  # 跳過標題
X, Y, Z = data[:,0], data[:,1], data[:,2]

# 3D 可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='r', marker='o', s=10)  # 可調顏色和大小

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Triangulated 3D Points')

plt.show()