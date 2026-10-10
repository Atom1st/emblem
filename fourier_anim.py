# -*- coding: utf-8 -*-
"""
fourier_anim.py
================
用傅里叶级数（一堆旋转的矢量 / epicycles）画出党徽轮廓的动画，ManimCE 版。

数据链路（同目录）：
    emblem1.png  --lunkuo.py-->  contour.npy     提取最外层轮廓
                 --caiyang2.py->  resampled.npy   等弧长重采样到 1024 点
                 --fft.py------>  fourier.npz    FFT，保留模长最大的 100 个谐波

本脚本只取模长最大的前 MAX_N 个（默认 50）个谐波。
实测：N=50 时与真实轮廓的平均偏差约为轮廓尺寸的 1%，肉眼基本看不出来，
但每帧要维护的几何量少了一半，动画更稳。

约定
----
* 图像坐标 y 向下，取复共轭后 y 向上，直接扔给 Manim 就是正着的党徽。
* 去掉 freq=0 的直流分量，曲线自带"质心在原点"的性质，于是旋转链条从
  ORIGIN 出发，笔尖位置 = 各矢量端点依次相加，不需要额外的整体平移。
* 路径点数固定为 SAMPLES，每帧只重算点集，不做 TracedPath 那种
  逐帧累积，长动画也不会越来越卡。

可调参数（也可以用环境变量临时覆盖）
------------------------------------
    $env:MAX_N   = 50     # 用几个本轮
    $env:SAMPLES = 900    # 每圈采样点数
    $env:DRAW    = 5.2    # 党徽在画面里的高度（Manim 单位）

渲染
----
    manim -ql fourier_anim.py EmblemFourier      #  854x480@15  快速预览
    manim -qm fourier_anim.py EmblemFourier      # 1280x720@30  中看
    manim -qh fourier_anim.py EmblemFourier      # 1920x1080@60 成片

    $env:NO_CARD=1; manim -qh fourier_anim.py EmblemFourier   # 掐掉字幕卡，给拼接用
"""

import os
from pathlib import Path

import numpy as np
from manim import *

config.background_color = BLACK

HERE = Path(__file__).resolve().parent     # 数据永远按脚本所在目录找，跟 cwd 无关
DATA_NPZ = HERE / "fourier.npz"
CONTOUR_NPY = HERE / "resampled.npy"

CN = "Microsoft YaHei"

PALETTE = [
    "#FF6B6B", "#FFA94D", "#FFD43B", "#51CF66",
    "#38D9A9", "#3BC9DB", "#4DABF7", "#748FFC",
    "#9775FA", "#DA77F2", "#F783AC", "#C2255C",
]

GHOST = "#7FD8FF"   # 真实轮廓
GOLD = "#FFC94A"    # 拟合出来的党徽

TRACE_OP = 0.95     # 已画出那段曲线的透明度
HEAD_OP = 0.5       # 笔尖高光的透明度

MAX_N = int(os.environ.get("MAX_N", 50))
SAMPLES = int(os.environ.get("SAMPLES", 900))
DRAW = float(os.environ.get("DRAW", 5.2))
NO_CARD = os.environ.get("NO_CARD", "0") == "1"   # 只渲到 hero 收尾，不渲字幕卡

# --------------------------------------------------------------------- 文案
# 所有字幕 / 标题都集中在这里，改字只需要动这一块。

TITLE_MAIN = "傅里叶级数"
TITLE_SUB = "任何闭合曲线，都是圆周运动的叠加"

CAP_ONE = "一根匀速旋转的矢量，笔尖画出一个圆"
TAG_ONE = "1 个本轮 = 1 个圆"

# 第二幕：(本轮数, 字幕, 画一圈的秒数)
STACK_STEPS = [
    (2, "加上第 2 根 —— 圆被拽变形了", 4.2),
    (4, "4 根 —— 轮廓开始长出折角", 4.6),
    (8, "8 根 —— 已经能认出是谁", 5.0),
    (16, "16 根 —— 高阶本轮在修细节", 5.4),
    (32, "32 根 —— 只剩细枝末节还没对上", 6.0),
]

CAP_RAMP = "本轮一根一根往上加，曲线边画边变"
CAP_QUIET = "辅助的圆退到幕后"
CAP_GHOST = "党徽原本的样子"
CAP_OVERLAY = f"再把 {MAX_N} 个本轮的结果叠上去"

CARD_TITLE = "傅里叶画党徽"
CARD_SUB = f"{MAX_N} 个旋转的向量，一圈不多，一圈不少"


# --------------------------------------------------------------------- 数据


def _load_npy(path, what):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"找不到 {what}：{path}\n"
            f"请先在 {HERE} 下依次跑 lunkuo.py -> caiyangang2.py -> fft.py 生成中间数据。"
        )
    return np.load(path)


def load_coefficients():
    """读 fourier.npz，按模长排序、去掉直流分量、共轭（翻转 y）、缩放。

    返回 (coeffs, freqs, scale)：
        coeffs[i] 已乘过 scale，直接累加即可得到画布坐标。
        scale 由完整 99 项的包围盒决定，保证党徽尺寸恒为 DRAW。
    """
    d = _load_npy(DATA_NPZ, "fourier.npz")
    c = d["coeffs"].astype(np.complex128)
    f = d["freqs"].astype(np.int64)

    order = np.argsort(-np.abs(c))
    c, f = c[order], f[order]

    keep = f != 0
    c, f = c[keep], f[keep]

    # 用全部非直流分量算包围盒，这样 scale 与 MAX_N 无关，改参数时画面不跳
    full = np.exp(1j * np.outer(np.arange(2048) / 2048.0, TAU * f)) @ c
    span = max(full.real.max() - full.real.min(), full.imag.max() - full.imag.min())
    scale = DRAW / span

    n = min(MAX_N, len(c))
    c, f = c[:n], f[:n]

    c = np.conj(c)                      # 图像 y 向下 -> 翻成 y 向上

    return c * scale, f, scale


def load_reference(samples, scale):
    """真实轮廓（resampled.npy），同样共轭 + 按质心归零 + 同一 scale。

    这样它和拟合曲线站在完全相同的坐标系里，误差数字才诚实。
    """
    pts = _load_npy(CONTOUR_NPY, "resampled.npy")[:, :2]
    if abs(pts[0, 0] - pts[-1, 0]) < 1e-9 and abs(pts[0, 1] - pts[-1, 1]) < 1e-9:
        pts = pts[:-1]

    z = np.conj(pts[:, 0] + 1j * pts[:, 1])
    z = z - z.mean()

    seg = np.abs(np.diff(np.concatenate([z, z[:1]])))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.arange(samples) / samples * s[-1]

    xr = np.interp(target, s, np.concatenate([z.real, z[:1].real]))
    yi = np.interp(target, s, np.concatenate([z.imag, z[:1].imag]))
    return (xr + 1j * yi) * scale


def to_points(z):
    return np.stack([z.real, z.imag, np.zeros(len(z))], axis=1)


# --------------------------------------------------------------------- 图形


class PathSlice(VMobject):
    """轨迹：显示整条路径的前缀（画出来的部分）或后缀（笔尖高光）。

    比 TracedPath 快的地方：点数恒定，每帧只是重新 set_points，
    不会随着时间越堆越多。
    """

    def __init__(self, color=GOLD, width=4.0, opacity=0.95, tail=0, **kwargs):
        super().__init__(**kwargs)
        self.set_fill(opacity=0)
        self.set_stroke(color, width=width, opacity=opacity)
        self.tail = tail
        self._pts = None
        self._key = None

    def load(self, pts):
        self._pts = np.asarray(pts, dtype=float)
        self._key = None
        self.goto(0)

    def goto(self, k):
        if self._pts is None:
            return
        k = int(k)
        if k == self._key:
            return
        self._key = k

        n = len(self._pts)
        k = max(0, min(k, n - 1))

        if self.tail > 0:
            a = max(0, k - self.tail)
            seg = self._pts[a:k + 1]
            if len(seg) < 2:
                seg = self._pts[max(0, k - 1):k + 1]
        else:
            seg = self._pts[:k + 1]
            if len(seg) < 2:
                seg = self._pts[:2]

        self.set_points_as_corners(seg)

    def clear(self):
        self._key = None


# --------------------------------------------------------------------- 场景


class EmblemFourier(MovingCameraScene):

    # ------------------------------------------------------------ 初始化

    def construct(self):
        self.coeffs, self.freqs, self.scale = load_coefficients()
        self.n_max = len(self.coeffs)

        # E[i, k] = e^{2 pi i f_k t_i}，一次性算好，之后每帧只是一次矩阵乘
        t = np.arange(SAMPLES) / float(SAMPLES)
        self.E = np.exp(1j * np.outer(t, TAU * self.freqs))

        self.reference = load_reference(SAMPLES, self.scale)
        self.errors = self._error_table()

        self.clock = ValueTracker(0.0)          # 一圈之内的进度 [0, 1]
        self.n_tracker = ValueTracker(1.0)     # 当前用了几个本轮（可为小数）
        self.dim = ValueTracker(1.0)           # 辅助圆的整体透明度

        self._last_count = -1
        self._last_dim = -1.0
        self._last_op = -1.0
        self._loaded_n = None

        self.radii = np.abs(self.coeffs)
        self.phases = np.angle(self.coeffs)

        self._build_grid()
        self._build_machine()
        self._build_pen()
        self._build_hud()

        self.plane.set_z_index(0)
        # 轨迹压在轮子下面：笔尖正好骑在最里面那个轮上，
        # 如果轨迹在上层，每个轮子都会被自己画过的那一段盖住。
        self.trace.set_z_index(1)
        self.head.set_z_index(2)
        self.machine.set_z_index(3)
        self.machine.add_updater(self._update_machine)
        self.pen.add_updater(self._update_pen)

        self.act1_open()
        self.act2_stack()
        self.act3_converge()
        self.act4_compare()
        self.act5_finale()

    # ------------------------------------------------------------ 构造元素

    def _build_grid(self):
        self.plane = NumberPlane(
            x_range=[-3.4, 3.4, 1],
            y_range=[-3.0, 3.0, 1],
            x_length=7.2,
            y_length=6.35,
            stroke_width=1,
            stroke_opacity=0.16,
            axis_config={"stroke_opacity": 0.32, "stroke_width": 1.6},
            background_line_style={"stroke_opacity": 0.11},
        )

    def _build_machine(self):
        colors = [PALETTE[i % len(PALETTE)] for i in range(self.n_max)]

        self.vectors = VGroup()
        self.circles = VGroup()
        for i in range(self.n_max):
            r = float(self.radii[i])
            self.vectors.add(
                Line(ORIGIN, [r, 0, 0], color=colors[i],
                     stroke_width=max(1.6, 3.6 - 0.04 * i), stroke_opacity=0)
            )
            self.circles.add(
                Circle(radius=r, color=colors[i],
                       stroke_width=max(1.4, 2.8 - 0.028 * i), stroke_opacity=0)
            )

        # 圆的形状建好之后不再变，先存一份「圆心在原点时」的点集，
        # 之后每帧写 home + 圆心 就是绝对定位。
        # 注意这里绝对不能用 shift()：那是相对位移，每帧累加会把圆甩出画面。
        self._home = [c.points.copy() for c in self.circles]

        self.machine = VGroup(self.vectors, self.circles)

    def _build_pen(self):
        self.trace = PathSlice(color=GOLD, width=3.4, opacity=TRACE_OP)
        self.head = PathSlice(color="#FFF3C4", width=6.0, opacity=HEAD_OP, tail=20)

        glow = Circle(radius=0.11, color=GOLD, fill_opacity=0.35, fill_color=GOLD,
                      stroke_width=0)
        dot = Dot(radius=0.042, color=WHITE)
        self.pen = VGroup(glow, dot)
        self.pen.set_z_index(20)

    def _build_hud(self):
        self.caption_grp = VGroup()

        self.n_num = DecimalNumber(1, num_decimal_places=0, font_size=28, color=GOLD)
        self.n_lab = VGroup(
            Text("本轮数量  N =", font=CN, font_size=20, color=GREY_B),
            self.n_num,
        ).arrange(RIGHT, buff=0.18)

        self.err_num = DecimalNumber(
            self.errors[1], num_decimal_places=3, font_size=28, color=GHOST
        )
        self.err_lab = VGroup(
            Text("平均误差", font=CN, font_size=20, color=GREY_B),
            self.err_num,
        ).arrange(RIGHT, buff=0.18)

        self.n_lab.to_corner(UL, buff=0.42)
        self.err_lab.to_corner(UR, buff=0.42)
        self.n_lab[1].align_to(self.err_lab[1], DOWN)
        self.readout = VGroup(self.n_lab, self.err_lab)

        # 读数不是单独动画出来的，而是每帧从 n_tracker 反推，
        # 所以「看到的圈数」和「写着的圈数 / 误差」在结构上就不可能对不上。
        self._shown_n = -1
        self._shown_err = -1.0
        self.n_num.add_updater(lambda m: self._sync_n(m))
        self.err_num.add_updater(lambda m: self._sync_err(m))

    def error_at(self, n):
        """当前本轮数（可取小数）对应的真实误差，在 errors 表里线性插值。"""
        n = float(np.clip(n, 1.0, float(self.n_max)))
        lo = int(np.floor(n))
        hi = min(lo + 1, self.n_max)
        f = n - lo
        return self.errors[lo] * (1.0 - f) + self.errors[hi] * f

    def _sync_n(self, mob):
        n = int(round(self.n_tracker.get_value()))
        if n != self._shown_n:
            mob.set_value(n)
            self._shown_n = n

    def _sync_err(self, mob):
        e = self.error_at(self.n_tracker.get_value())
        if abs(e - self._shown_err) > 5e-4:
            mob.set_value(e)
            self._shown_err = e

    # ------------------------------------------------------------ 每帧更新

    def _weights(self, n):
        """把"用了 n 个本轮"翻译成权重向量，支持小数（用于淡入新的一根）。"""
        n = float(np.clip(n, 1.0, self.n_max))
        k = int(np.floor(n))
        w = np.zeros(self.n_max)
        w[:k] = 1.0
        if k < self.n_max:
            w[k] = n - k
        return w

    def _path(self, n):
        return self.E @ (self.coeffs * self._weights(n))

    def _update_machine(self, mob, dt):
        n = self.n_tracker.get_value()
        theta = (self.clock.get_value() % 1.0) * TAU

        count = int(np.ceil(n - 1e-6))
        count = max(1, min(count, self.n_max))

        ang = self.phases[:count] + self.freqs[:count] * theta
        step = self.radii[:count, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)

        knots = np.zeros((count + 1, 3))
        knots[1:, :2] = np.cumsum(step, axis=0)
        ends = np.stack([knots[:-1], knots[1:]], axis=1)   # (count, 2, 3)

        # 直接写点集，别用 put_start_and_end_on / move_to：
        # 那两个要走 get_critical_point + rotate + shift，50 根要 7 ms，
        # 每帧预算直接爆掉；这里整套只要 0.7 ms。
        for i in range(count):
            self.vectors[i].set_points_as_corners(ends[i])
            self.circles[i].set_points(self._home[i] + knots[i])

        dim = self.dim.get_value()
        idx = np.arange(count)
        appear = np.clip(n - idx, 0.0, 1.0)      # 小数 N 时，最后一根正在淡入
        falloff = 1.0 - 0.50 * idx / max(count - 1, 1)

        if count != self._last_count or abs(dim - self._last_dim) > 2e-3:
            for i in range(count):
                self.vectors[i].set_stroke(opacity=float(appear[i] * dim * falloff[i]))
                self.circles[i].set_stroke(opacity=float(appear[i] * dim))
            for i in range(count, self.n_max):
                self.vectors[i].set_stroke(opacity=0.0)
                self.circles[i].set_stroke(opacity=0.0)
            self._last_count = count
            self._last_dim = dim
            self._last_op = float(appear[count - 1] * dim)
        else:
            # 只有最新那一根的透明度还在变，补它就行，不用重刷全部 100 个
            op = float(appear[count - 1] * dim)
            if abs(op - self._last_op) > 2e-3:
                last = count - 1
                self.vectors[last].set_stroke(opacity=op * float(falloff[last]))
                self.circles[last].set_stroke(opacity=op)
                self._last_op = op

    def _progress(self):
        """一圈之内的进度 [0, 1]；刚好转完一圈时算作 1，曲线留在屏幕上。"""
        c = self.clock.get_value()
        p = c - np.floor(c)
        if p < 1e-9 and c > 1e-9:
            p = 1.0
        return p

    def _update_pen(self, mob, dt):
        n = self.n_tracker.get_value()
        k = min(int(self._progress() * SAMPLES), SAMPLES - 1)

        z = self._path(n)
        if n != self._loaded_n:
            # 本轮数变了，笔尖走的路也变了，整条轨迹重新装载
            self._loaded_n = n
            pts = to_points(z)
            self.trace.load(pts)
            self.head.load(pts)

        self.trace.goto(k)
        self.head.goto(k)
        mob.move_to([z[k].real, z[k].imag, 0.0])

    # ------------------------------------------------------------ 小工具

    def rewind(self):
        """回到起点并清掉轨迹。"""
        self.clock.set_value(0.0)
        self.trace.clear()
        self.head.clear()

    def spin(self, turns=1.0, run_time=5.0, reset=True):
        """匀速转 turns 圈。"""
        if reset:
            self.rewind()
        self.play(
            self.clock.animate.increment_value(turns),
            run_time=run_time,
            rate_func=linear,
        )

    def say(self, text, n=None, run_time=0.8, color=GREY_B, size=23):
        """换字幕；如果给了 n，顺便把本轮数也换到 n。

        顺序是「旧字幕淡出 + N 一起变 → 新字幕淡入」，
        这样字幕上写「8 根」的时候，画面上确实就是 8 个圈。
        """
        new = Text(text, font=CN, font_size=size, color=color)
        new.to_edge(DOWN, buff=0.34)
        old = self.caption_grp
        self.caption_grp = VGroup(new)

        half = run_time / 2
        anims = []
        if len(old) > 0:
            anims.append(FadeOut(old[0], shift=0.15 * UP))
        if n is not None:
            anims.append(self.n_tracker.animate.set_value(n))

        if anims:
            self.play(*anims, run_time=half, rate_func=smooth)
        self.play(FadeIn(new, shift=0.15 * UP), run_time=half)
        return new

    def _error_table(self):
        # 拟合曲线是倒着画出来的（取共轭的副作用），所以反转一下再和真实轮廓对齐
        ref = self.reference
        out = {}
        for k in range(1, self.n_max + 1):
            approx = (self.E[:, :k] @ self.coeffs[:k])[::-1]
            out[k] = float(np.abs(approx - ref).mean())
        return out

    # ------------------------------------------------------------ 第一幕

    def act1_open(self):
        self.next_section("open")

        t1 = Text(TITLE_MAIN, font=CN, font_size=48, color=WHITE)
        t2 = Text(TITLE_SUB, font=CN, font_size=24, color=GREY_B)
        card = VGroup(t1, t2).arrange(DOWN, buff=0.45)

        self.play(FadeIn(t1, shift=0.3 * UP), run_time=1.1)
        self.play(FadeIn(t2, shift=0.2 * UP), run_time=0.8)
        self.wait(1.0)
        self.play(FadeOut(card, shift=0.3 * UP), run_time=0.8)

        # 标题走完再把机器放进来，避免开场字被圆压住
        self.add(self.machine, self.trace, self.head, self.pen)
        self.play(
            FadeIn(self.plane, shift=0.2 * DOWN),
            FadeIn(self.readout),
            run_time=1.0,
        )

        self.say(CAP_ONE, run_time=0.6)
        self.spin(1.0, run_time=5.0)

        tag = Text(TAG_ONE, font=CN, font_size=21, color=YELLOW)
        tag.next_to(self.vectors[0], UP, buff=0.3)
        self.play(FadeIn(tag, shift=0.2 * DOWN), run_time=0.7)
        self.spin(0.55, run_time=2.6, reset=False)
        self.play(FadeOut(tag), run_time=0.4)

    # ------------------------------------------------------------ 第二幕

    def act2_stack(self):
        self.next_section("stack")

        for n, text, rt in STACK_STEPS:
            self.say(text, n=n)
            self.spin(1.0, run_time=rt)

    # ------------------------------------------------------------ 第三幕

    def act3_converge(self):
        self.next_section("converge")

        self.say(CAP_RAMP, n=32)

        # 一整圈里把 N 从 32 平滑推到 MAX_N。
        # 笔尖一路往前不回头，已经画出来的那一段也跟着一起变形，
        # 所以不会出现「同一段弧线反复重画」的卡顿感。
        self.rewind()
        self.play(
            self.clock.animate.increment_value(1.0),
            self.n_tracker.animate.set_value(MAX_N),
            run_time=10.0,
            rate_func=linear,
        )

    # ------------------------------------------------------------ 第四幕

    def act4_compare(self):
        self.next_section("compare")

        # 先把上一幕画完的金色党徽收起来，只留真实轮廓，避免两条线打架
        self.say(CAP_QUIET)
        self.play(
            FadeOut(self.readout),
            self.dim.animate.set_value(0.10),
            self.trace.animate.set_stroke(opacity=0.0),
            self.head.animate.set_stroke(opacity=0.0),
            run_time=1.6,
        )

        self.next_section("ghost")
        self.say(CAP_GHOST, run_time=0.6)
        self.ghost = VMobject().set_points_as_corners(to_points(self.reference))
        self.ghost.set_stroke(GHOST, width=2.6, opacity=0.55)
        self.add(self.ghost)
        self.play(Create(self.ghost), run_time=3.2)
        self.wait(0.6)

        self.next_section("overlay")
        self.say(CAP_OVERLAY, run_time=0.8)
        self.play(
            self.trace.animate.set_stroke(opacity=TRACE_OP),
            self.head.animate.set_stroke(opacity=HEAD_OP),
            run_time=0.6,
        )
        self.spin(1.0, run_time=8.0)

        self.say(f"平均偏差 {self.errors[MAX_N]:.3f}", run_time=0.8, color=GOLD, size=24)
        self.wait(1.4)

    # ------------------------------------------------------------ 第五幕

    def act5_finale(self):
        self.next_section("finale")

        self.play(
            FadeOut(self.caption_grp),
            FadeOut(self.ghost),
            run_time=1.0,
        )

        glow = VMobject().set_points_as_corners(to_points(self._path(MAX_N)))
        glow.set_stroke(GOLD, width=24, opacity=0.10)
        glow.set_z_index(4)

        self.play(
            self.dim.animate.set_value(0.0),
            FadeIn(glow),
            self.camera.frame.animate.scale(0.88),
            run_time=2.0,
        )

        self.machine.clear_updaters()
        self.pen.clear_updaters()
        self.n_num.clear_updaters()
        self.err_num.clear_updaters()

        self.play(
            AnimationGroup(
                FadeOut(self.machine),
                FadeOut(self.pen),
                FadeOut(self.plane),
                lag_ratio=0,
            ),
            run_time=1.4,
        )
        self.remove(self.head)
        self.wait(2.2)

        # ---- 收尾卡片：党徽淡出，字幕单独留在黑场上 ----
        # NO_CARD=1 时只渲到 hero 收尾，字幕卡留给 formula_swarm.py 之后单独渲，
        # 这样拼接时不用靠 -ss 去猜切点（manim 会复用前五幕的缓存）。
        # 字幕卡的起始时间由动画本身报出来，拼接时按这个数切。
        print("CARD_START = %.3f s" % self.time)
        if NO_CARD:
            return

        self.next_section("card")
        self.play(
            FadeOut(self.trace),
            FadeOut(glow),
            self.camera.frame.animate.scale(1 / 0.88),
            run_time=1.6,
        )

        title = Text(CARD_TITLE, font=CN, font_size=46, color=GOLD)
        sub = Text(CARD_SUB, font=CN, font_size=23, color=GREY_B)
        formula = MathTex(r"z(t)=\sum_{k}c_k\,e^{2\pi i kt}", font_size=46, color=WHITE)
        card = VGroup(title, sub, formula).arrange(DOWN, buff=0.6)

        self.play(FadeIn(title, shift=0.25 * UP), run_time=1.0)
        self.play(FadeIn(sub, shift=0.2 * UP), run_time=0.8)
        self.play(Write(formula), run_time=1.6)
        self.wait(2.6)

        self.play(FadeOut(card, shift=0.2 * DOWN), run_time=1.6)
        self.wait(0.4)
