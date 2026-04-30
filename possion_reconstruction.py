import numpy as np
import scipy.sparse as sp
from scipy import integrate
from skimage import measure # 用於最後的 Marching Cubes
import open3d as o3d


# 定義 1D B-spline
def B_spline_1D(t):
    """一維 B-spline 基底函數 (緊支撐於 [-1.5, 1.5])"""
    t = abs(t)
    if t < 0.5:
        return 0.75 - t ** 2
    elif t < 1.5:
        return 0.5 * (1.5 - t) ** 2
    return 0.0


def B_spline_1D_deriv(t):
    """一維 B-spline 的導數"""
    if -0.5 < t < 0.5:
        return -2.0 * t
    elif 0.5 <= t < 1.5:
        return -(1.5 - t)
    elif -1.5 < t <= -0.5:
        return (1.5 + t)
    return 0.0


def evaluate_basis_value(node, q):
    """
    計算節點 i 的基函數在空間點 q 的值 F_i(q)
    node: 八叉樹節點，包含中心 node.c 和寬度 node.w
    q: 待求值的 3D 空間點 (x, y, z)
    """
    # 1. 計算點 q 相對於節點中心的正規化距離
    # 將世界座標偏移到以節點中心為原點，並依照寬度縮放
    normalize_x = (q[0] - node.c[0]) / node.w
    normalize_y = (q[1] - node.c[1]) / node.w
    normalize_z = (q[2] - node.c[2]) / node.w

    # 2. 分別計算三個維度上的 B-spline 值
    val_x = B_spline_1D(normalize_x)
    val_y = B_spline_1D(normalize_y)
    val_z = B_spline_1D(normalize_z)

    # 3. 三者相乘得到 3D 空間的貢獻值
    # 除以寬度的三次方 (體積歸一化)
    # 確保不同深度的節點在積分時具有一致的能量權重
    norm_factor = 1.0 / (node.w ** 3)

    return (val_x * val_y * val_z) * norm_factor

#  定義 3D 節點的梯度函數
def evaluate_basis_gradient(node, q):
    """計算節點基函數 F_o 在空間點 q 的梯度 ∇F_o(q)"""
    # 正規化座標：將世界座標 q 映射到節點的局部空間
    dx = (q[0] - node.c[0]) / node.w
    dy = (q[1] - node.c[1]) / node.w
    dz = (q[2] - node.c[2]) / node.w

    # 3D 函數值是三個 1D 函數的乘積，根據連鎖律求偏導
    # ∂F/∂x = B'(dx) * B(dy) * B(dz) / node.w
    grad_x = B_spline_1D_deriv(dx) * B_spline_1D(dy) * B_spline_1D(dz) / node.w
    grad_y = B_spline_1D(dx) * B_spline_1D_deriv(dy) * B_spline_1D(dz) / node.w
    grad_z = B_spline_1D(dx) * B_spline_1D(dy) * B_spline_1D_deriv(dz) / node.w

    # 記得除以體積歸一化因子 (1 / node.w^3)
    norm_factor = 1.0 / (node.w ** 3)
    return np.array([grad_x, grad_y, grad_z]) * norm_factor


# 計算重疊積分
def compute_laplacian_inner_product(node_i, node_j):
    """
    計算 A_ij = - ∫ ∇F_i(q) · ∇F_j(q) dq
    """
    # 這裡用數值積分。為了效率，積分範圍只取兩個節點重疊的包圍盒
    overlap_box = get_overlap_bounding_box(node_i, node_j)

    if overlap_box_is_empty(overlap_box):
        return 0.0  # 沒有重疊，積分為 0 (這是保證稀疏性的關鍵)

    # 定義被積函數：兩個梯度的點積
    def integrand(z, y, x):
        q = np.array([x, y, z])
        grad_i = evaluate_basis_gradient(node_i, q)
        grad_j = evaluate_basis_gradient(node_j, q)
        return -np.dot(grad_i, grad_j)

    # 在重疊區域內進行 3D 數值積分 (scipy.integrate)
    result, _ = integrate.tplquad(
        integrand,
        overlap_box.x_min, overlap_box.x_max,
        lambda x: overlap_box.y_min, lambda x: overlap_box.y_max,
        lambda x, y: overlap_box.z_min, lambda x, y: overlap_box.z_max
    )
    return result


#一個結構來存儲節點的信息，包括它的邊界、中心、深度以及它的子節點。
class OctreeNode:
    def __init__(self, center, width, depth, node_id):
        self.c = center        # 節點中心 (x, y, z)
        self.w = width         # 節點寬度
        self.depth = depth     # 目前深度
        self.idx = node_id     # 用於 Ax=b 矩陣的索引，唯一編號
        #每個 x_i 就對應一個八叉樹節點的函數值
        self.children = []     # 儲存 8 個子節點
        self.is_leaf = True    # 是否為葉子節點
        self.points_indices = [] # 落在該節點內的點雲索引

    def is_empty(self):
        return len(self.points_indices) == 0


def get_nodes_in_support(q, root):
    """
    從根節點開始，遞迴尋找所有支撐域包含點 q 的節點
    q: 查詢點 (x, y, z)
    root: 八叉樹的根節點
    """
    nodes_found = []

    def search(node):
        if node is None:
            return

        # 1. 計算點 q 到節點中心的距離
        dist = np.abs(q - node.c)

        # 2. 判斷點 q 是否在該節點的支撐域內 (1.5w 是邊界)
        # 注意：對於不同階數的 B-spline，這個係數可能不同 (例如 1階是 1.0w)
        support_limit = 1.5 * node.w

        if np.all(dist < support_limit):
            # 如果是葉子節點且點在範圍內，這就是我們要找的「基函數」
            # 且只有有點雲貢獻的節點(有 idx)才參與計算
            if node.is_leaf and node.idx != -1:
                nodes_found.append(node)
            # 即使當前節點包含 q，我們仍要檢查它的子節點
            # 因為子節點更小，它們的基函數可能也覆蓋了 q
            for child in node.children:
                search(child)

    search(root)
    return nodes_found

def build_adaptive_octree(points, max_depth=8):
    # 1. 計算點雲的整體包圍盒 (Bounding Box),建立 Root
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    # 為了方便，我們取最大邊長作為正方體邊長
    width = np.max(max_bound - min_bound)
    #給每個葉節點分配唯一的 index（ID）
    node_counter = [0]  # 使用 list 來傳遞引用，追蹤總節點數
    root = OctreeNode(center, width, depth=0, node_id=node_counter[0])
    root.points_indices = list(range(len(points)))  # 初始時所有點都在根節點

    leaf_nodes = []

    # 2. 開始遞歸細分
    def subdivide(node):
        # 停止條件：達到最大深度 或 節點內已經沒有點（或者點數太少）
        if node.depth >= max_depth or len(node.points_indices) <= 1:
        #標記葉節點:每個 leaf node 都會變成： 一個「離散單元」
        # 之後對應 Ax=b 裡的一個 unknown
            node.idx = node_counter[0]
            # index +1
            node_counter[0] += 1
            leaf_nodes.append(node)
            return
        #備切 8 個子空間:父 cube → 切成 8 個 cube
        node.is_leaf = False
        child_w = node.w / 2
        offset = child_w / 2

        # 定義 8 個子節點相對於父節點中心的偏移
        offsets = [
            (-offset, -offset, -offset), (offset, -offset, -offset),
            (-offset, offset, -offset), (offset, offset, -offset),
            (-offset, -offset, offset), (offset, -offset, offset),
            (-offset, offset, offset), (offset, offset, offset)
        ]

        # 建立 8 個子節點
        child_points_buckets = [[] for _ in range(8)]

        # 將父節點的點分配給 8 個子節點 (Spatial Partitioning)
        #用 bit 編碼空間位置
        for pt_idx in node.points_indices:
            pt = points[pt_idx]
            # 判斷點在哪個象限(判斷點在哪個子 cube)
            child_idx = 0
            if pt[0] > node.c[0]: child_idx += 1# x方向
            if pt[1] > node.c[1]: child_idx += 2# y方向
            if pt[2] > node.c[2]: child_idx += 4 # z方向
            child_points_buckets[child_idx].append(pt_idx)

        # 遞迴建立有意義的子節點
        for i in range(8):
            if len(child_points_buckets[i]) > 0:  # 只細分有點存在的空間 (Adaptive)
                child_center = node.c + np.array(offsets[i])
                child_node = OctreeNode(child_center, child_w, node.depth + 1, -1)
                child_node.points_indices = child_points_buckets[i]
                node.children.append(child_node)
                subdivide(child_node)
            else:
                node.children.append(None)  # 空節點不參與後續計算

    subdivide(root)
    return leaf_nodes  # 返回所有參與計算的葉子節點

# 重建連續的指示函數 χ(q)
def evaluate_chi(q):
    """
    給定空間中任意一點 q，算出該點是在物體內還是物體外。
    公式： χ(q) = Σ x_i * F_i(q)
    """
    chi_value = 0.0
    # 同樣的，只要找 q 附近的節點就好，遠處的節點 F_i(q) = 0
    nearby_nodes = get_nodes_in_support(q, nodes)
    for node in nearby_nodes:
        # F_i(q) = B(dx) * B(dy) * B(dz) / node.w^3
        F_val = evaluate_basis_value(node, q)

        # 累加：該節點的權重 * 該節點對 q 的影響力
        chi_value += x_coefficients[node.idx] * F_val

    return chi_value

# 1. 讀取點雲
pcd = o3d.io.read_point_cloud("fused_output.ply")
points = np.asarray(pcd.points)  # 形狀 (N, 3)
normals = np.asarray(pcd.normals)  # 形狀 (N, 3)

# 2. 建構八叉樹 (這裡假設你有一個函數可以劃分空間)
# nodes 是一個包含所有八叉樹葉子節點的列表
nodes = build_adaptive_octree(points, max_depth=8)
num_nodes = len(nodes)

#3， 初始化稀疏矩陣 A 和 向量 b
# 使用 LIL 格式方便逐個元素賦值，後續求解時轉為 CSR 格式
A = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
b = np.zeros(num_nodes, dtype=np.float32)


# 構建觀測向量 b (投影點雲法向量)
# b_j = < ∇·V, F_j > = - ∫ V · ∇F_j dq
# 離散化近似為： b_j = - Σ (法向量_s) · ∇F_j(點_s)
for s_idx, (p_s, n_s) in enumerate(zip(points, normals)):
    # 為了效率，我們只找包含該點的節點及其相鄰節點（因為 F_j 離開這裡就為 0）
    active_nodes = get_nodes_in_support(p_s, nodes)
    for node_j in active_nodes:
        j = node_j.idx
        # 1. 計算基函數 j 在點 p_s 處的梯度
        grad_F = evaluate_basis_gradient(node_j, p_s)
        # 2. 向量內積並累加到 b_j
        b[j] -= np.dot(n_s, grad_F)

# 構建拉普拉斯矩陣 A
# A_ij = < ΔF_i, F_j >
# ==========================================
for i, node_i in enumerate(nodes):
    # 由於 F_i 是緊支撐的，它只跟它的鄰居節點有重疊
    # 所以我們只需要遍歷它的鄰居，保證了 A 的極度稀疏性
    for j_idx in node_i.neighbors:
        node_j = nodes[j_idx]

        # 計算重疊積分
        inner_prod = compute_laplacian_inner_product(node_i, node_j)

        # 賦值給矩陣，A 是對稱的
        A[i, j_idx] = inner_prod


# 解方程 Ax = b ，x 陣列包含了所有節點的權重係數
A_csr = A.tocsr()
x_coefficients, _ = sp.linalg.cg(A_csr, b, tol=1e-5)


#4. Marching Cubes 提取表面 (終局)
# 4.1. 建立一個均勻的 3D 網格空間 (用來評估場的值)
# 假設我們的物體落在 [0, 1] 範圍，解析度設為 128x128x128
grid_resolution = 128
x_grid = np.linspace(0, 1, grid_resolution)
y_grid = np.linspace(0, 1, grid_resolution)
z_grid = np.linspace(0, 1, grid_resolution)

# 4.2. 評估網格上每個點的 χ 值，生成 3D 體素數據 (Volume Data)
volume_data = np.zeros((grid_resolution, grid_resolution, grid_resolution))

for ix in range(grid_resolution):
    for iy in range(grid_resolution):
        for iz in range(grid_resolution):
            q = np.array([x_grid[ix], y_grid[iy], z_grid[iz]])
            volume_data[ix, iy, iz] = evaluate_chi(q)

# 4.3. 使用 Marching Cubes 提取等值面
# 尋找數值等於 0.5 的交界處 (物體表面)
vertices, faces, normals, values = measure.marching_cubes(volume_data, level=0.5)

# 5. 輸出模型！(可以存成 OBJ 或 PLY)
print(f"成功生成了 {len(vertices)} 個頂點和 {len(faces)} 個三角形！")
save_to_obj("reconstructed_mesh.obj", vertices, faces)