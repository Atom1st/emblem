"""
donghua.py
ManimCE 傅里叶本轮动画 —— 3b1b 风格演示版（重写版 v2）

分幕:
  1. 标题 + 一根本轮画圆
  2. 逐个加入本轮, 复平面网格 + 半径标注
  3. 拖动 N, 观察收敛
  4. 揭示目标轮廓(幽灵线) + 收敛误差
  5. 完整绘制 + 总结卡片

修复:
  - clock 取模, 避免 theta 无限增长
  - trace 每圈清空, 避免轨迹叠加成乱麻
  - 只更新可见本轮, 避免每帧遍历 100 个 mobject
  - 缓存 opacity, 避免每帧 set_stroke
"""

import numpy as np
from manim import *

CN = "Microsoft YaHei"
MONO = "Consolas"

PALETTE = [
    "#FF6B6B", "#FFA94D", "#FFD43B", "#51CF66",
    "#38D9A9", "#3BC9DB", "#4DABF7", "#748FFC",
    "#9775FA", "#DA77F2", "#F783AC", "#C2255C",
]

GHOST = "#7FD8FF"


class DanghuiFourier(MovingCameraScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_vectors = 100
        self.n_visible_circles = 10
        self.cycle_seconds = 20
        self.draw_radius = 6.0
        self._data = None
        self._n_coeffs = 0

        # 缓存上一帧的可见性与透明度, 避免每帧重复 set_stroke
        self._last_count = -1
        self._last_opacity = -1.0

    # ------------------------------------------------------------------ 数据

    def load(self):
        if self._data is not None:
            return self._data

        d = np.load("fourier.npz")
        coeffs = d["coeffs"].astype(np.complex128)
        freqs = d["freqs"].astype(np.int64)

        order = np.argsort(-np.abs(coeffs))
        coeffs, freqs = coeffs[order], freqs[order]
        keep = freqs != 0
        coeffs, freqs = coeffs[keep], freqs[keep]
        coeffs = np.conj(coeffs)

        total = np.abs(coeffs).sum()
        coeffs = coeffs * (self.draw_radius / total if total > 0 else 1.0)

        radii = np.abs(coeffs)
        phases = np.angle(coeffs)
        self._data = {"coeffs": coeffs, "freqs": freqs, "radii": radii, "phases": phases}
        self._n_coeffs = len(coeffs)
        return self._data

    def pen_path(self, n=None, samples=1200):
        d = self.load()
        c, f = d["coeffs"], d["freqs"]
        if n is not None:
            k = max(1, min(int(n), len(c)))
            c, f = c[:k], f[:k]
        t = np.linspace(0, 1, samples, endpoint=False)
        z = np.sum(c[None, :] * np.exp(1j * np.outer(t, 2 * np.pi * f)), axis=1)
        return z

    # ------------------------------------------------------------------ 构造

    def construct(self):
        self.load()
        self.n = min(self.n_vectors, self._n_coeffs)

        # clock 只在 [0, 1) 内循环, 走完一圈归零
        self.clock = ValueTracker(0)
        self.clock.add_updater(lambda m, dt: m.increment_value(dt / self.cycle_seconds))

        self.n_tracker = ValueTracker(1.0)
        self.dim = ValueTracker(1.0)

        self.vectors, self.circles = self.build_vectors()

        self.pen = self.build_pen()
        pen = self.pen

        self.trace = TracedPath(
            pen.get_center,
            stroke_color=YELLOW,
            stroke_width=3.0,
            stroke_opacity=0.9,
        )
        self.trail = TracedPath(
            pen.get_center,
            stroke_color=YELLOW,
            stroke_width=2.0,
            stroke_opacity=0.35,
            dissipating_time=1.2,
        )

        # 先手动调一次, 保证 pen 在加入场景前就位于正确位置
        self.update_pen(pen, 0)
        pen.add_updater(self.update_pen)

        pen.set_z_index(10)
        self.trace.set_z_index(5)
        self.trail.set_z_index(4)
        self.add(self.vectors, self.circles, self.trail, self.trace, pen)

        self.plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=8.6,
            y_length=7.5,
            stroke_width=1,
            stroke_opacity=0.14,
            axis_config={"stroke_opacity": 0.3, "stroke_width": 1.4},
            background_line_style={"stroke_opacity": 0.12},
        )

        self.titles = VGroup()
        self.body = VGroup()

        self.act_1_open()
        self.act_2_build()
        self.act_3_converge()
        self.act_4_reveal()
        self.act_5_finish()

    # -------------------------------------------------------------- 视觉元素

    def build_vectors(self):
        d = self.load()
        vectors, circles = VGroup(), VGroup()
        for i, (r, ph, fr) in enumerate(zip(d["radii"], d["phases"], d["freqs"])):
            color = PALETTE[i % len(PALETTE)]
            vec = Line(ORIGIN, [r, 0, 0], color=color, stroke_width=2.0)
            cir = Circle(radius=r, color=color, stroke_width=1.8)
            vectors.add(vec)
            circles.add(cir)
        return vectors, circles

    def build_pen(self):
        glow = Circle(radius=0.1, color=YELLOW, fill_opacity=0.4, stroke_width=0)
        dot = Dot(radius=0.035, color=WHITE)
        return VGroup(glow, dot)

    def update_vectors(self, mob, dt):
        """每帧更新本轮几何 + 透明度。

        优化点:
          - theta 取模, 避免无限增长
          - 只遍历 count 个, 不再遍历剩余 100 个
          - opacity 用缓存, 变化小于阈值时跳过 set_stroke
        """
        d = self.load()
        total = len(d["radii"])
        N = self.n_tracker.get_value()
        count = min(int(N) + 1, total)
        theta = (self.clock.get_value() % 1.0) * TAU
        dim = self.dim.get_value()

        current = ORIGIN
        for i in range(count):
            r, ph, fr = d["radii"][i], d["phases"][i], d["freqs"][i]
            angle = ph + fr * theta
            end = current + r * np.array([np.cos(angle), np.sin(angle), 0.0])
            self.vectors[i].put_start_and_end_on(current, end)
            self.circles[i].move_to(current)
            current = end

        # 透明度单独一轮: 只有 count 变化或 dim 变化明显时才更新
        if count != self._last_count or abs(dim - self._last_opacity) > 0.01:
            for i in range(total):
                if i < count:
                    appear = max(0.0, min(1.0, N - i))
                    fade = (1.0 - i / total * 0.55) * appear * dim
                else:
                    fade = 0.0
                self.vectors[i].set_stroke(opacity=fade)
                self.circles[i].set_stroke(opacity=fade, width=1.8 - i * 0.004)
            self._last_count = count
            self._last_opacity = dim

        self.pen_pos = current

    def update_pen(self, mob, dt):
        old_clock = self.clock.get_value()
        self.update_vectors(mob, dt)
        mob.move_to(self.pen_pos)

        # 跨过一个整周期 -> 清空 trace, 避免轨迹叠加
        if int(self.clock.get_value()) > int(old_clock):
            self.trace.clear_points()

    def set_visible(self, n, run_time=1.5, **kw):
        return self.play(
            self.n_tracker.animate.set_value(n),
            run_time=run_time,
            rate_func=smooth,
            **kw,
        )

    def title(self, text, sub=None, **kw):
        t = Text(text, font=CN, font_size=34, color=WHITE, **kw)
        t.to_edge(UP, buff=0.35)
        if sub:
            s = Text(sub, font=CN, font_size=19, color=GREY_B).next_to(t, DOWN, buff=0.12)
            t = VGroup(t, s)
        self.titles.add(t)
        return t

    def caption(self, text, **kw):
        c = Text(text, font=CN, font_size=22, color=GREY_B, **kw)
        c.to_edge(DOWN, buff=0.4)
        self.body.add(c)
        return c

    def counter(self, **kw):
        num = DecimalNumber(0, num_decimal_places=0, font_size=26, color=YELLOW)
        lab = Text("个本轮", font=CN, font_size=20, color=GREY_B).next_to(num, RIGHT, buff=0.15)
        g = VGroup(num, lab).next_to(self.plane, RIGHT, buff=0.35)
        self.body.add(g)
        return num, g

    def dim_rings(self, to=0.0, run_time=1.2, **kw):
        return self.play(self.dim.animate.set_value(to), run_time=run_time, **kw)

    def draw_one_cycle(self, run_time=10, clear_trace=True):
        """画一整圈, 然后 clock 归零 + 清 trace。"""
        if clear_trace:
            self.trace.clear_points()
        self.play(self.clock.animate.increment_value(1), run_time=run_time, rate_func=linear)
        self.clock.set_value(0)
        if clear_trace:
            self.trace.clear_points()

    # ------------------------------------------------------------------ 第一幕

    def act_1_open(self):
        self.next_section("opening")

        head = self.title("任何闭合曲线\n都是圆周运动的叠加")
        sub = self.caption("傅里叶级数 · 几何直观")

        self.play(
            FadeIn(self.plane, shift=0.2 * DOWN),
            FadeIn(head, shift=0.3 * DOWN),
            FadeIn(sub),
            run_time=1.6,
        )

        # 一根本轮 -> 一个圆
        self.play(self.clock.animate.set_value(1), run_time=6, rate_func=linear)
        self.clock.set_value(0)

        lab = Text("一根本轮 = 一个圆", font=CN, font_size=22, color=YELLOW)
        lab.next_to(self.vectors[1], UP, buff=0.25)
        self.body.add(lab)
        self.play(FadeIn(lab, shift=0.2 * DOWN), run_time=0.8)
        self.play(self.clock.animate.increment_value(0.5), run_time=3, rate_func=linear)

    # ------------------------------------------------------------------ 第二幕

    def act_2_build(self):
        self.next_section("build")

        self.play(FadeOut(self.titles), FadeOut(self.body), run_time=0.8)

        c1 = self.caption("加上第二根 -> 尖底花瓣")
        self.play(FadeIn(c1), run_time=0.4)
        self.set_visible(2, run_time=1.5)
        self.play(self.clock.animate.increment_value(1), run_time=6, rate_func=linear)

        c2 = self.caption("第三根、第四根... 曲线的细节逐渐浮现")
        self.play(FadeOut(c1, shift=0.2 * UP), FadeIn(c2), run_time=1.2)
        self.set_visible(4, run_time=2.0)
        self.play(self.clock.animate.increment_value(1), run_time=7, rate_func=linear)

        c3 = self.caption(f"高阶本轮半径很小，只画前 {self.n_visible_circles} 个")
        self.play(FadeOut(c2, shift=0.2 * UP), FadeIn(c3), run_time=1.2)
        self.set_visible(self.n_visible_circles, run_time=2.0)
        self.play(self.clock.animate.increment_value(1), run_time=7, rate_func=linear)

    # ------------------------------------------------------------------ 第三幕

    def act_3_converge(self):
        self.next_section("converge")

        self.play(FadeOut(self.body), run_time=0.7)

        self.dim.set_value(1.0)
        self.set_visible(3, run_time=1.0)
        self.play(self.clock.animate.increment_value(1), run_time=4, rate_func=linear)

        t = self.title("把本轮一个个加回去")
        num, grp = self.counter()
        self.play(FadeIn(t), FadeIn(grp), run_time=0.9)

        for n in (6, 12, 25, 45, 70):
            self.play(
                self.n_tracker.animate.set_value(n),
                num.animate.set_value(n),
                run_time=1.1,
                rate_func=smooth,
            )
            self.play(self.clock.animate.increment_value(0.12), run_time=1.6, rate_func=linear)

        self.play(
            self.n_tracker.animate.set_value(self.n),
            num.animate.set_value(self.n),
            run_time=2.0,
        )
        self.draw_one_cycle(run_time=10)

    # ------------------------------------------------------------------ 第四幕

    def act_4_reveal(self):
        self.next_section("reveal")

        self.play(FadeOut(self.body), run_time=1.0)
        self.dim_rings(0.0, run_time=1.5)

        self.target = self.make_target()
        self.play(Create(self.target), run_time=3.0)
        self.play(self.target.animate.set_stroke(opacity=0.4), run_time=1.0)

        self.play(
            self.camera.frame.animate.scale(0.42).move_to(ORIGIN),
            FadeOut(self.plane),
            run_time=2.5,
        )

        err = self.error_label()
        self.play(FadeIn(err, shift=0.2 * UP), run_time=1.0)

        self.draw_one_cycle(run_time=10)

    def make_target(self, samples=900):
        z = self.pen_path(None, samples=samples)
        pts = np.stack([z.real, z.imag, np.zeros_like(z.real)], axis=1)
        m = VMobject().set_points_as_corners(pts)
        m.set_stroke(GHOST, width=2.2, opacity=0.4)
        return m

    def error_label(self):
        z_approx = self.pen_path(self.n, samples=900)
        z_full = self.pen_path(None, samples=900)
        err = np.abs(z_approx - z_full).mean()

        val = Text(f"{err:.3f}", font=MONO, font_size=26, color=YELLOW)
        lab = Text("平均误差", font=CN, font_size=20, color=GREY_B).next_to(val, RIGHT, buff=0.18)
        grp = VGroup(val, lab).to_edge(UP, buff=0.35)
        self.body.add(grp)
        return grp

    # ------------------------------------------------------------------ 第五幕

    def act_5_finish(self):
        self.next_section("finish")

        self.play(FadeOut(self.body), FadeOut(self.target), run_time=1.0)
        self.play(
            self.camera.frame.animate.scale(1 / 0.42).move_to(ORIGIN),
            run_time=2.0,
        )

        self.dim.set_value(1.0)
        self.set_visible(self.n, run_time=1.5)
        self.dim_rings(0.12, run_time=1.2)
        self.dim_rings(1.0, run_time=1.2)
        self.draw_one_cycle(run_time=11)

        self.pen.clear_updaters()
        self.clock.clear_updaters()
        # 先清掉 trace/trail 的 points, 避免 FadeOut 时 family zip 错位
        self.trace.clear_points()
        self.trail.clear_points()

        self.play(
            AnimationGroup(
                FadeOut(self.vectors),
                FadeOut(self.circles),
                FadeOut(self.plane),
                FadeOut(self.pen),
                lag_ratio=0,
            ),
            run_time=1.4,
        )
        self.remove(self.trace, self.trail)
        self.remove(self.clock)

        card = VGroup(
            Text("傅里叶变换", font=CN, font_size=44, color=WHITE),
            Text(
                f"{self.n} 个旋转的向量\n画出党徽的每一点",
                font=CN,
                font_size=24,
                color=GREY_B,
                line_spacing=0.9,
            ),
            MathTex(r"z(t)=\sum_{k}c_k\,e^{2\pi i kt}", font_size=48, color=YELLOW),
        ).arrange(DOWN, buff=0.7)

        self.play(FadeIn(card[0], shift=0.3 * UP), run_time=1.0)
        self.play(FadeIn(card[1], shift=0.2 * UP), run_time=0.8)
        self.play(Write(card[2]), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(card), run_time=1.5)
        self.wait(0.5)
