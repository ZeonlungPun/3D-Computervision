import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_colmap_dense_bin(path):
    """
    读取 COLMAP dense depth/normal .bin 文件
    格式一般为:
        width&height&channels& + float32 binary data
    返回:
        data: ndarray, shape = (H, W) 或 (H, W, C)
    """
    path = Path(path)
    with open(path, "rb") as f:
        # 读取 header: width&height&channels&
        def read_until_ampersand():
            chars = []
            while True:
                c = f.read(1)
                if c == b'&':
                    break
                if c == b'':
                    raise EOFError("Unexpected end of file while reading header.")
                chars.append(c.decode("utf-8"))
            return int("".join(chars))

        width = read_until_ampersand()
        height = read_until_ampersand()
        channels = read_until_ampersand()

        # 读取 float32 数据
        data = np.fromfile(f, dtype=np.float32)

    expected = width * height * channels
    if data.size != expected:
        raise ValueError(
            f"Data size mismatch: got {data.size}, expected {expected} "
            f"(w={width}, h={height}, c={channels})"
        )

    data = data.reshape((width, height, channels), order="F")
    data = np.transpose(data, (1, 0, 2))  # -> (H, W, C)

    if channels == 1:
        data = data[:, :, 0]  # -> (H, W)

    return data


def visualize_depth(depth, title="Depth Map", save_path=None):
    """
    可视化深度图
    """
    if depth.ndim == 3:
        # 如果不是单通道，默认取第一个通道
        depth = depth[:, :, 0]

    # 去掉无效值
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("No valid depth values found.")

    dmin = np.percentile(depth[valid], 2)
    dmax = np.percentile(depth[valid], 98)

    plt.figure(figsize=(10, 8))
    plt.imshow(depth, cmap="plasma", vmin=dmin, vmax=dmax)
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


if __name__ == "__main__":
    bin_path = "/home/zonekey/project/colmap/auto_reconstruction/dense/0/stereo/depth_maps/viff.030.ppm.geometric.bin"   # 改成你的 bin 文件路径
    depth = read_colmap_dense_bin(bin_path)
    visualize_depth(depth, title=Path(bin_path).name)