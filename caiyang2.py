"""
resample.py
把 contour.npy 按弧长均匀重采样到 N 个点。

用法:
    python resample.py contour.npy 1024 resampled.npy
"""

import sys
import numpy as np


def resample_closed(pts, n):
    # 去掉重复的闭合末点
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    # 计算累积弧长
    d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1]

    # 目标等弧长位置
    target = np.linspace(0, total, n, endpoint=False)

    # 线性插值
    x = np.interp(target, s, np.concatenate([pts[:, 0], pts[:1, 0]]))
    y = np.interp(target, s, np.concatenate([pts[:, 1], pts[:1, 1]]))

    out = np.stack([x, y], axis=1)
    # 闭合
    out = np.vstack([out, out[0]])
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "contour.npy"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    dst = sys.argv[3] if len(sys.argv) > 3 else "resampled.npy"

    pts = np.load(src)
    out = resample_closed(pts, n)
    np.save(dst, out)
    print(f"{len(pts)} 点 -> {len(out)} 点，已保存 {dst}")
