# -*- coding: utf-8 -*-
"""
formula_swarm.py
=================
成千上万个公式飞过来，汇聚成党徽。

为什么不用 manim 画
-------------------
manim 的 Camera.display_image_mobject 每帧要给**每一个** ImageMobject 新建一张
整屏 1920x1080 的 RGBA 图，再做一次整屏 alpha 合成。实测单个的固定开销就有
13.5 ms，5000 个就是 67 秒一帧，这条路是死的。

所以这里只用 manim 做一次性素材（MathTex 截图），动画本体是一个 numpy 粒子
系统：所有粒子画进同一张 1080p RGB 缓冲，帧直接管道进 ffmpeg。实测 8000
粒子的合成开销约 32 ms/帧。

画面里那些公式是真的
--------------------
resampled.npy 的 DFT 一共有 1024 个复系数（fourier.npz 只留了模长最大的
100 个）。这里把 1024 项全摆出来，每个系数在多个景深上重复出现凑成几千个
粒子。字幕说的「党徽 = 1024 个复系数之和」是实话。

时间轴
------
    0.0 - 5.0 s   涌入    按景深分层从画面外螺旋收进来
    5.0 - 8.0 s   盘旋    绕党徽涡旋，半径收紧
    8.0 - 14.5 s  汇聚    逐粒子错峰 lerp 到目标点（先描轮廓，再填内部）
    14.5 - 17.5 s 融合    粒子缩小淡出，金色党徽淡入

用法
----
    python formula_swarm.py
    $env:SWARM_N=2000 ; python formula_swarm.py     # 粗剪
    $env:SWARM_N=0    ; python formula_swarm.py     # 只看收尾那一拍

环境变量：SWARM_N 粒子数 / SWARM_W,SWARM_H 分辨率 / SWARM_FPS 帧率
          SWARM_OUT 输出路径 / SWARM_SEED 随机种子
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

W = int(os.environ.get("SWARM_W", 1920))
H = int(os.environ.get("SWARM_H", 1080))
FPS = int(os.environ.get("SWARM_FPS", 60))
N = int(os.environ.get("SWARM_N", 8000))
SEED = int(os.environ.get("SWARM_SEED", 20240607))
OUT = os.environ.get("SWARM_OUT", str(HERE / "media" / "videos" / "formula_swarm" / "swarm.mp4"))

GOLD = (255, 201, 74)          # 和主片同一支金
GOLD_DIM = (150, 118, 44)      # 远景粒子用的小号暗版

# ---------------------------------------------------------------- 时间轴

T_IN = 3.5          # 涌入（要冲得快，不然前两秒画面中间是空的）
T_SWIRL = 4.5       # 盘旋
CONV_START = 8.0    # 汇聚开始
CONV_DUR = 3.4      # 单个粒子的汇聚时长
MERGE = 3.0         # 融合

# ---------------------------------------------------------------- 构图

CX, CY = W / 2.0, H / 2.0
HALF_DIAG = float(np.hypot(W / 2.0, H / 2.0))
PXU = H / 8.0                       # 1 个 manim 单位 = H/8 像素（1080p 时 135）
OUT_SCALE = 1.00                    # 党徽相对主片尺寸的缩放
R_EMBLEM = PXU * 5.2 * OUT_SCALE / 2.0   # 党徽半径（像素）

FRAC_OUTLINE = 0.045                # 少量大卡片 -> 党徽轮廓是认得出字的一圈
FRAC_FILL = 0.955                   # 大量小卡片 -> 内部的金色织体

# 内部织体的亮度按「到党徽边界的距离」分四档。
# np.maximum 不会累加亮度，几千张卡片叠上去取的是最大值，所以想要层次
# 只能把层次预先烤进素材、再按位置分配 —— 贴边最亮、深入内部最暗，
# 这样党徽天然是「亮边 + 琥珀色内核」，而不是一块死金。
FILL_TOKENS = [
    "∂", "Σ", "∫", "∇", "√", "π", "ω", "φ", "θ", "λ", "μ", "σ", "Δ", "∞",
    "≈", "≠", "±", "→", "⇒", "∈", "ℝ", "ℂ",
    "c", "k", "z", "t", "n", "e", "f", "Z", "1/n", "|z|", "Re", "Im",
    "cos", "sin", "d", "α", "β", "γ",
]

FILL_TIERS = [(158, 124, 43), (116, 90, 30), (80, 62, 21), (50, 39, 14)]
FILL_EDGES = [0.20, 0.36, 0.54]     # 归一化深度的分档阈值（对着实际分布取的）

# 织体字形比大卡片小 0.0068 倍画幅宽
FILL_SIZE_K = 0.0068

# 盘旋段光靠一个环，中间会空出一个大洞。加一层占 24% 的内圈粒子专门填中间，
# 它们汇聚时照常飞向自己的目标点，所以只影响涌入 / 盘旋两段。
INNER_FRAC = 0.24
INNER_R = 0.40
INNER_OMEGA = 1.7

# 收尾：粒子原地缩小溶解，党徽在底下淡入
SHRINK_TIME = 1.6
SHRINK_FLOOR = 0.22


def ease_out_cubic(u):
    return 1.0 - (1.0 - u) ** 3


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# ================================================================= 素材


def _find_font():
    names = ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"]
    roots = [Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
             Path("/Library/Fonts"), Path.home() / ".fonts"]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return str(p)
    raise RuntimeError("找不到等宽字体")


def crop_ink(arr, thresh=30):
    """裁到墨迹包围盒，返回连续 uint8 RGB。"""
    mask = arr.max(axis=2) > thresh
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.ascontiguousarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def coefficient_cards():
    """1024 个复系数 -> 2048 张系数卡片（PIL 渲染，约 0.5 s）。

    这里重跑一次 DFT，把 fourier.npz 丢掉的那 924 项也拿回来。
    """
    pts = np.load(HERE / "resampled.npy")[:, :2]
    if abs(pts[0, 0] - pts[-1, 0]) < 1e-9 and abs(pts[0, 1] - pts[-1, 1]) < 1e-9:
        pts = pts[:-1]

    z = pts[:, 0] + 1j * pts[:, 1]
    n = len(z)
    Z = np.fft.fft(z) / n
    k = np.fft.fftfreq(n, d=1.0 / n)

    order = np.argsort(-np.abs(Z))            # 大系数排在前面
    Z, k = Z[order], k[order]

    # 字号跟画幅宽度走成比例，这样 480p 粗剪看到的密度和 1080p 成片一致
    size = max(11, int(round(W * 0.0135)))
    font = ImageFont.truetype(_find_font(), size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    texts = []
    for c, kk in zip(Z, k):
        mag, ang = abs(c), np.angle(c)
        # 两行：上行模长、下行相位。相位是让每一项独一无二的东西——
        # 只印 |c| 的话会有好几张都是 .0207。
        # 换行而不是接在后面，是为了保持卡片宽度不变（宽度决定轮廓上甩边多少）。
        texts.append("c[%+d]=%.4f\ne^(%+.3fi)" % (int(kk), mag, ang))
        texts.append("f[%+d]=%+d\n|c|=%.4f" % (int(kk), int(kk), mag))

    # 画布按最长的一行来定，别浪费像素；两行 -> 高度加倍
    cw = int(max(probe.textlength(t.split("\n")[0], font=font) for t in texts)) + 10
    ch = int(size * 3.3) + 6

    out = []
    for text in texts:
        img = Image.new("RGB", (cw, ch), (0, 0, 0))
        d = ImageDraw.Draw(img)
        for li, line in enumerate(text.split("\n")):
            d.text((4, 2 + li * int(size * 1.55)), line, font=font, fill=GOLD)
        s = crop_ink(np.array(img))
        if s is not None:
            out.append(s)
    return out



DECOR = [
    r"\sum_{k}", r"\int_0^1", r"\oint", r"\prod_{k}",
    r"e^{2\pi i kt}", r"e^{i\theta}", r"\frac{1}{n}", r"\frac{\partial z}{\partial t}",
    r"\partial", r"\nabla", r"\Delta", r"\sqrt{n}", r"\infty", r"\omega",
    r"\phi", r"\theta", r"\sin", r"\cos", r"\Re", r"\Im",
    r"|c_k|", r"\|\cdot\|", r"2\pi", r"\pi i", r"\approx", r"\to",
    r"\equiv", r"\arg\max", r"\cdots", r"\mathrm{d}t", r"\hat{c}_k",
    r"k = 0", r"N = 1024", r"z(t)", r"c_k", r"f_k", r"\mathbb{R}^2",
    r"\cos 2\pi k t", r"\sin 2\pi k t", r"|Z_k|", r"\mathrm{arg}",
    r"\left(\sum\right)", r"e^{-\,i\omega t}", r"\frac{\mathrm{d}f}{\mathrm{d}t}",
]


def decor_formulas():
    """用 manim MathTex 渲染装饰公式，Camera 截图后阈值化成硬边 sprite。

    没有 LaTeX 就退回空列表，不影响主流程。
    """
    import manim as M

    M.config.background_color = M.BLACK
    font_size = max(14, int(round(W * 0.0156)))
    out, failed = [], 0
    for tex_src in DECOR:
        try:
            tex = M.MathTex(tex_src, font_size=font_size, color="#%02x%02x%02x" % GOLD)
            # 每条公式都要用全新的 Camera：capture_mobjects 是在现有 pixel_array
            # 上继续画的，不清空的话 44 张会叠成同一张图（全是糊的）。
            cam = M.Camera(pixel_width=1000, pixel_height=240,
                           background_color=M.BLACK)
            cam.capture_mobjects([tex])
            s = crop_ink(cam.pixel_array[:, :, :3].copy(), thresh=40)
            if s is None:
                continue
            if s.shape[1] > 420:                       # 太宽的等比缩一下
                s = cv2.resize(s, (420, max(10, int(s.shape[0] * 420 / s.shape[1]))),
                               interpolation=cv2.INTER_AREA)
            out.append(s)
        except Exception as exc:
            failed += 1
            if failed == 1:
                print("      ! 第一条装饰公式就失败了，后面跳过：", exc)
    return out


FILL_N_TILES = int(os.environ.get("SWARM_TILES", 512))


def glyph_tiles():
    """内部织体用的短字形小块。

    长系数卡片缩到 30 px 以下就是一团糊；短字形（∂ Σ ∇ π ω …）本身就像
    「织体」，缩到十几像素依然看得出是数学符号，密铺出来的质感才对。
    """
    size = max(7, int(round(W * FILL_SIZE_K)))
    font = ImageFont.truetype(_find_font(), size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rng = np.random.default_rng(SEED + 7)

    cw = int(probe.textlength(" ".join(FILL_TOKENS[:6]), font=font)) + 10
    ch = int(size * 1.7) + 4

    out = []
    for _ in range(FILL_N_TILES):
        k = int(rng.integers(2, 6))
        text = " ".join(FILL_TOKENS[i] for i in
                        rng.integers(0, len(FILL_TOKENS), k))
        img = Image.new("RGB", (cw, ch), (0, 0, 0))
        ImageDraw.Draw(img).text((3, 2), text, font=font, fill=GOLD)
        s = crop_ink(np.array(img))
        if s is not None:
            out.append(s)
    return out


def shrink(arr, scale, tint=None):
    h, w = arr.shape[:2]
    nh, nw = max(5, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    if tint is not None:
        f = np.asarray(tint, np.float32) / 255.0
        small = np.clip(small.astype(np.float32) * f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(small)


def build_atlas():
    """返回 (atlas, n_big, n_tile)。

    屏幕上能塞下多少「公式墨水」是有上限的：1920x1080 一屏 2.07 M 像素，
    几千个粒子如果每张都按可读字号画，墨水会超出版面十几倍，整个屏幕糊成
    一坨金。所以分两种素材：

        大卡片  只给轮廓那几百个粒子 -> 汇聚后党徽轮廓是能认得出字的一圈
        短字形  给内部那几千个粒子   -> 密铺成金色织体，边缘能看到笔笔纹理

    内部再按到边界的距离分四档亮度（亮边 -> 暗芯）。每档内部靠景深做
    「缩小 + 压暗」的远景版，深度感直接烤进素材，运行时零成本。
    """
    print("[1/4] 渲染系数卡片 ...", flush=True)
    t0 = time.perf_counter()
    cards = coefficient_cards()
    print("      %d 张，%.1f s" % (len(cards), time.perf_counter() - t0), flush=True)

    print("[2/4] 渲染装饰公式 + 织体字形 ...", flush=True)
    t0 = time.perf_counter()
    deco = decor_formulas()
    tiles = glyph_tiles()
    print("      装饰 %d + 字形 %d，%.1f s"
          % (len(deco), len(tiles), time.perf_counter() - t0), flush=True)

    big = cards + deco
    if not big or not tiles:
        raise RuntimeError("素材没做出来")

    # 0/1 = 大卡片近/远（轮廓）；2..5 = 织体字形四档亮度（内部）
    atlas = [big, [shrink(s, 0.55, GOLD_DIM) for s in big]]
    for tint in FILL_TIERS:
        atlas.append([shrink(s, 1.0, tint) for s in tiles])

    mb = sum(s.nbytes for tier in atlas for s in tier) / 1e6
    a, b = big[0], tiles[0]
    print("      图集 %d 档，%.0f MB；大卡片 %dx%d，字形块 %dx%d"
          % (len(atlas), mb, a.shape[1], a.shape[0], b.shape[1], b.shape[0]),
          flush=True)
    return atlas, len(big), len(tiles)


# ================================================================= 目标点


def outline_targets(count):
    """沿傅里叶路径等弧长取点（画布单位 -> 像素），并把党徽摆正到画面正中。"""
    import fourier_anim as FA

    coeffs, freqs, _ = FA.load_coefficients()
    S = 6000
    t = np.arange(S) / S
    z = np.exp(1j * np.outer(t, FA.TAU * freqs)) @ coeffs

    seg = np.abs(np.diff(np.concatenate([z, z[:1]])))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tg = np.linspace(0.0, s[-1], count, endpoint=False)
    xi = np.interp(tg, s, np.concatenate([z.real, z[:1].real]))
    yi = np.interp(tg, s, np.concatenate([z.imag, z[:1].imag]))

    # 沿法向抖一点，别描成一根头发丝
    rng = np.random.default_rng(SEED + 1)
    dx, dy = np.gradient(xi), np.gradient(yi)
    ln = np.hypot(dx, dy) + 1e-12
    j = rng.normal(0.0, 0.055, count)
    xi = xi + (dy / ln) * j
    yi = yi - (dx / ln) * j

    # manim 的 y 向上，图像 y 向下，所以这里取负
    px = xi * PXU * OUT_SCALE
    py = -yi * PXU * OUT_SCALE

    # 平移：党徽包围盒中心落在画面正中（傅里叶路径的质心不在包围盒中心）
    px += CX - 0.5 * (px.min() + px.max())
    py += CY - 0.5 * (py.min() + py.max())
    return px, py


def emblem_masks():
    """返回 (填充用内部点, 整幅金框党徽 RGBA)。"""
    g = cv2.imread(str(HERE / "emblem1.png"), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError("找不到 emblem1.png")
    bw = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)[1]
    return bw


def fill_targets(count, bw, anchor):
    """从党徽内部采样，并返回每个点的「深入程度」0..1（0 = 贴边）。

    和轮廓批次对齐：等高 + 包围盒中心重合。
    """
    inner = cv2.erode(bw, np.ones((17, 17), np.uint8))
    ys, xs = np.nonzero(inner)
    if len(xs) == 0:
        ys, xs = np.nonzero(bw)

    ys_all, xs_all = np.nonzero(bw)
    by0, by1 = ys_all.min(), ys_all.max()
    bx0, bx1 = xs_all.min(), xs_all.max()

    k = anchor[2] / (by1 - by0)
    ox = anchor[0] - 0.5 * (bx0 + bx1) * k
    oy = anchor[1] - 0.5 * (by0 + by1) * k

    # 距离变换 -> 每个采样点离党徽边界有多远，用来分亮度档
    dist = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dmax = float(dist.max()) or 1.0

    rng = np.random.default_rng(SEED + 2)
    pick = rng.integers(0, len(xs), count)
    px = xs[pick].astype(np.float64) * k + ox
    py = ys[pick].astype(np.float64) * k + oy
    depth_in = np.clip(dist[ys[pick], xs[pick]] / dmax, 0.0, 1.0)
    return px, py, depth_in


def emblem_alpha(bw, anchor, height):
    """整幅画布大小的党徽 alpha 通道（0..1），带一圈外发光。

    缩放比例和 fill_targets 用的是同一个 bbox 公式，保证填充粒子和这张
    图严丝合缝地对在一起。
    """
    ys, xs = np.nonzero(bw)
    by0, by1, bx0, bx1 = ys.min(), ys.max(), xs.min(), xs.max()

    k = height / (by1 - by0)
    w_px = max(2, int(round((bx1 - bx0) * k)))
    h_px = max(2, int(round((by1 - by0) * k)))

    sub = (bw[by0:by1 + 1, bx0:bx1 + 1] > 0).astype(np.float32)
    small = cv2.resize(sub, (w_px, h_px), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((H, W), np.float32)
    y0 = int(round(anchor[1] - h_px / 2.0))
    x0 = int(round(anchor[0] - w_px / 2.0))
    yy0, xx0 = max(0, y0), max(0, x0)
    yy1, xx1 = min(H, y0 + h_px), min(W, x0 + w_px)
    canvas[yy0:yy1, xx0:xx1] = small[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0]

    glow = cv2.GaussianBlur(canvas, (0, 0), max(6.0, height * 0.022))
    return np.clip(canvas + glow * 0.85, 0.0, 1.0)


# ================================================================= 动画


def particle_state(count, n_outline, px, py, depth_in, n_big, n_tile):
    rng = np.random.default_rng(SEED)

    depth = rng.random(count).astype(np.float32)
    ang0 = rng.random(count).astype(np.float32) * (2 * np.pi)
    omega = (0.16 + 0.44 * depth).astype(np.float32)          # 近的转得快
    ease_p = (2.2 + 2.8 * rng.random(count)).astype(np.float32)

    # 起始半径贴着画幅外沿，别甩太远——否则前两秒粒子都在画面外，
    # 只剩左右两边挤成一堵字墙。
    r_orbit = (1.30 + 0.20 * depth) * R_EMBLEM
    r_start = HALF_DIAG * (0.98 + 0.32 * depth)

    # 四分之一塞进内圈，把盘旋段中间那个洞填掉
    inner = rng.random(count) < INNER_FRAC
    r_orbit = np.where(inner, r_orbit * (INNER_R + 0.12 * depth), r_orbit)
    omega = np.where(inner, omega * INNER_OMEGA, omega).astype(np.float32)

    is_far = depth <= 0.45                                     # 轮廓粒子用缩小压暗档
    is_outline = np.arange(count) < n_outline                  # 先描轮廓

    # 汇聚错峰：轮廓先到（按角度扫一圈），内部晚一拍
    d_out = rng.random(n_outline) * 1.1
    d_fill = d_out[rng.integers(0, n_outline, count - n_outline)] \
        + 1.2 + rng.random(count - n_outline) * 1.0
    delay = np.concatenate([d_out, d_fill]).astype(np.float32)

    # 0/1 = 大卡片（轮廓）近/远；2..5 = 织体字形按深入程度分四档亮度
    fill_tier = 2 + np.digitize(depth_in, FILL_EDGES)
    tier = np.where(is_outline, np.where(is_far, 1, 0), fill_tier).astype(np.int32)

    return dict(depth=depth, ang0=ang0, omega=omega, ease_p=ease_p,
                r_orbit=r_orbit, r_start=r_start, tier=tier, delay=delay,
                spr_big=rng.integers(0, n_big, count).astype(np.int32),
                spr_tile=rng.integers(0, n_tile, count).astype(np.int32),
                px=px, py=py)


def r_orbit_swirl(st, t):
    v = np.clip((t - T_IN) / T_SWIRL, 0.0, 1.0)
    return st["r_orbit"] * (1.0 - 0.42 * smoothstep(v))


def positions(st, t):
    """向量化算出所有粒子在 t 秒时的像素坐标。"""
    ang = st["ang0"] + st["omega"] * t

    # 涌入：每个粒子用不同的 ease 指数 -> 自然错峰，但都在 T_IN 时刻归位
    u = np.clip(t / T_IN, 0.0, 1.0)
    g = 1.0 - (1.0 - u) ** st["ease_p"]
    r_in = st["r_start"] + (st["r_orbit"] - st["r_start"]) * g

    # 盘旋：半径收紧，角度继续转
    r = np.where(t <= T_IN, r_in, r_orbit_swirl(st, t))
    x_base = CX + r * np.cos(ang)
    y_base = CY + r * np.sin(ang)

    # 汇聚：从 CONV_START 时刻的轨道位姿插值到目标点
    r0 = r_orbit_swirl(st, CONV_START)
    a0 = st["ang0"] + st["omega"] * CONV_START
    x0 = CX + r0 * np.cos(a0)
    y0 = CY + r0 * np.sin(a0)

    e = ease_out_cubic(np.clip((t - CONV_START - st["delay"]) / CONV_DUR, 0.0, 1.0))
    live = t > CONV_START
    x = np.where(live, x0 + (st["px"] - x0) * e, x_base)
    y = np.where(live, y0 + (st["py"] - y0) * e, y_base)
    return x, y


def sprite_view(src, s):
    """按比例 s 从中心裁出一个视图（numpy 视图，零拷贝）。

    用来做「溶解」：粒子不换档、不淡出，只是原地缩小。比每帧 resize 便宜
    两个数量级（8000 张卡各 resize 一次要 80 ms，裁视图不要钱）。
    """
    if s >= 0.995:
        return src
    h, w = src.shape[:2]
    nh, nw = max(3, int(h * s)), max(6, int(w * s))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return src[y0:y0 + nh, x0:x0 + nw]


def composite(buf, x, y, tier, spr_big, spr_tile, atlas, scale=1.0):
    h, w = buf.shape[:2]
    for i in range(len(x)):
        t = tier[i]
        src = sprite_view(atlas[t][spr_big[i] if t < 2 else spr_tile[i]], scale)
        sh, sw = src.shape[:2]
        # 粒子坐标是「中心」，落笔要减掉半个 sprite
        xi = int(x[i]) - (sw >> 1)
        yi = int(y[i]) - (sh >> 1)
        if xi < 0 or yi < 0 or xi + sw > w or yi + sh > h:
            continue
        np.maximum(buf[yi:yi + sh, xi:xi + sw], src, out=buf[yi:yi + sh, xi:xi + sw])


def render():
    if N <= 0:
        print("SWARM_N=0，只渲收尾那一拍")
        return render_merge_only()

    atlas, n_big, n_tile = build_atlas()

    print("[3/4] 采样目标点 ...", flush=True)
    n_out = max(24, int(N * FRAC_OUTLINE))
    ox, oy = outline_targets(n_out)
    anchor = (CX, CY, oy.max() - oy.min())
    fx, fy, fin = fill_targets(N - n_out, emblem_masks(), anchor)

    px = np.concatenate([ox, fx])
    py = np.concatenate([oy, fy])
    depth_in = np.concatenate([np.zeros(n_out), fin])
    st = particle_state(N, n_out, px, py, depth_in, n_big, n_tile)
    print("      轮廓 %d + 内部 %d，党徽 %.0f px 高" % (n_out, N - n_out, anchor[2]),
          flush=True)

    conv_end = CONV_START + float(st["delay"].max()) + CONV_DUR
    total = conv_end + MERGE
    nframes = int(round(total * FPS))

    alpha = emblem_alpha(emblem_masks(), anchor, anchor[2])
    gold_rgb = np.asarray(GOLD, np.float32)
    tmp = np.empty((H, W, 3), np.uint8)        # 党徽底图层的中转

    print("[4/4] 合成 %d 粒子 / %.1f s / %d 帧 ..." % (N, total, nframes), flush=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
           "-pix_fmt", "yuv420p", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    buf = np.zeros((H, W, 3), np.uint8)
    t_start = time.perf_counter()

    # 底层党徽比所有织体字形都亮（全都是 GOLD 的暗化版），所以粒子缩小到
    # 一定程度会被 np.maximum 自动「吃掉」——不用第二块缓冲，也不用整幅淡出。
    t_in = conv_end - 0.5

    for f in range(nframes):
        t = f / FPS

        # 1) 底图层：金色党徽（从黑起淡入）
        if t >= t_in:
            k = smoothstep((t - t_in) / (MERGE * 0.9))
            if k > 0.001:
                np.multiply(gold_rgb, (alpha * k)[:, :, None],
                            out=tmp, casting="unsafe")
                np.copyto(buf, tmp, casting="unsafe")
            else:
                buf[:] = 0
        else:
            buf[:] = 0

        # 2) 粒子层：原地缩小，不换档
        s = 1.0
        if t > conv_end:
            s = 1.0 - (1.0 - SHRINK_FLOOR) * smoothstep((t - conv_end) / SHRINK_TIME)
            if s <= 0.02:
                s = 0.0
        if s > 0.0:
            x, y = positions(st, t)
            composite(buf, x, y, st["tier"], st["spr_big"], st["spr_tile"],
                      atlas, scale=s)

        proc.stdin.write(buf.tobytes())

        if f % FPS == 0:
            done = f / nframes
            el = time.perf_counter() - t_start
            print("      %5.1f%%  t=%5.2fs  已用 %4.0fs  预计还要 %4.0fs"
                  % (done * 100, t, el, el / max(done, 1e-6) - el), flush=True)

    proc.stdin.close()
    proc.wait()
    print("完成 -> %s  (%.0f s)" % (OUT, time.perf_counter() - t_start))


def render_merge_only():
    """SWARM_N=0 的降级路径：只验证「金色党徽淡入」这一拍。"""
    total = 4.0
    nframes = int(total * FPS)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)

    h_px = PXU * 5.2 * OUT_SCALE
    anchor = (CX, CY, h_px)
    alpha = emblem_alpha(emblem_masks(), anchor, h_px)
    gold_rgb = np.asarray(GOLD, np.float32)

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    buf = np.zeros((H, W, 3), np.uint8)

    for f in range(nframes):
        k = smoothstep(f / (nframes - 1.0))
        buf[:] = 0
        tmp = (alpha * k)[:, :, None] * gold_rgb
        np.maximum(buf, tmp.astype(np.uint8), out=buf)
        proc.stdin.write(buf.tobytes())

    proc.stdin.close()
    proc.wait()
    print("完成 -> %s" % OUT)


if __name__ == "__main__":
    render()
