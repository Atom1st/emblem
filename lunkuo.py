"""
extract_danghui.py
针对纯黑白党徽图提取外轮廓。

用法:
    python extract_danghui.py danghui_bw.png
输出:
    contour.npy    # (N, 2) 单条闭合轮廓
    contour.json   # 同上
    preview.png    # 预览
"""

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract(image_path, min_area=500, epsilon=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    print("shape:", img.shape, "unique values:", np.unique(img)[:10])

    # 纯黑白图：>127 为白（党徽前景），<=127 为黑（背景）
    # 用 THRESH_BINARY 让白色变成 255，黑色变 0
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 如果上面反了（党徽变黑），把 THRESH_BINARY 换成 THRESH_BINARY_INV

    # 小去噪
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 只取最外层轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    contours.sort(key=cv2.contourArea, reverse=True)

    if not contours:
        raise RuntimeError("没找到轮廓")

    c = contours[0]
    print("最大轮廓原始点数:", len(c), "面积:", cv2.contourArea(c))

    approx = cv2.approxPolyDP(c, epsilon, True)
    pts = approx[:, 0, :].astype(float)

    # 闭合
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    return pts, binary


def preview(pts, binary, out="preview.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(binary, cmap="gray")
    axes[0].set_title("Binary Mask")
    axes[1].plot(pts[:, 0], pts[:, 1], "-", linewidth=1.0, color="red")
    axes[1].plot(pts[0, 0], pts[0, 1], "go", markersize=6, label="start")
    axes[1].invert_yaxis()
    axes[1].axis("equal")
    axes[1].legend()
    axes[1].set_title(f"Contour ({len(pts)} points)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"预览 -> {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "emblem1.png"
    pts, binary = extract(path)
    np.save("contour.npy", pts)
    with open("contour.json", "w") as f:
        json.dump({"closed": True, "points": pts.tolist()}, f)
    print(f"最终点数: {len(pts)}")
    preview(pts, binary)
