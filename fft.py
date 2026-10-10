"""
fft.py
对重采样后的轮廓做 FFT，最多保留指定数量的圆盘/谐波。

用法:
    python fft.py 8
    python fft.py 8 resampled.npy fourier.npz
"""

import sys
import numpy as np


def decompose(pts, num_harmonics=8):
    # 去掉闭合重复末点
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    z = pts[:, 0] + 1j * pts[:, 1]
    N = len(z)

    Z = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)

    # 按模长从大到小排序（先画大的圆）
    order = np.argsort(-np.abs(Z))

    # 只保留前 num_harmonics 个
    order = order[:num_harmonics]

    return {
        "coeffs": Z[order],
        "freqs": freqs[order],
        "N": N,
        "num_harmonics": len(order),
    }


if __name__ == "__main__":
    # 第一个参数：最多几个圆盘/谐波
    num_harmonics = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    src = sys.argv[2] if len(sys.argv) > 2 else "resampled.npy"
    dst = sys.argv[3] if len(sys.argv) > 3 else "fourier.npz"

    pts = np.load(src)
    result = decompose(pts, num_harmonics=num_harmonics)
    np.savez(dst, **result)

    print(f"num_harmonics = {result['num_harmonics']} (N={result['N']})")
    print(f"保存 {len(result['coeffs'])} 个频率分量 -> {dst}")
    print("前 5 个分量（模长）:", np.abs(result["coeffs"][:5]))
