"""
ML Math Lab — Backend ของแอปประยุกต์
รวม 3 หัวข้อ: Calculus (Derivative/Integral/Gradient Descent),
Sigmoid/Softmax, และ Linear Algebra (Matrix/Eigen/Linear Transform)
"""
import math
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X = sp.Symbol("x", real=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCALS = {
    "x": X, "pi": sp.pi, "e": sp.E,
    "ln": sp.log, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "tanh": sp.tanh,
    "sigmoid": lambda t: 1 / (1 + sp.exp(-t)),
}
_FORBIDDEN = ("__", "import", "lambda", "exec", "eval", "open", "os.", "sys.")


# ------------------------------------------------------------------ Calculus
def parse_function(expr_str: str) -> sp.Expr:
    """แปลงข้อความ เช่น 'x^3 - 3x' เป็นนิพจน์ SymPy (ตัวแปรเดียวคือ x)"""
    if not expr_str or not expr_str.strip():
        raise ValueError("กรุณากรอกฟังก์ชัน f(x)")
    if len(expr_str) > 120 or any(w in expr_str.lower() for w in _FORBIDDEN):
        raise ValueError("รูปแบบฟังก์ชันไม่ถูกต้อง")
    try:
        expr = parse_expr(expr_str, local_dict=_LOCALS, transformations=_TRANSFORMS)
    except Exception as exc:
        raise ValueError(f"อ่านฟังก์ชันไม่ได้: {exc}") from None
    extra = expr.free_symbols - {X}
    if extra:
        names = ", ".join(sorted(str(s) for s in extra))
        raise ValueError(f"ใช้ตัวแปรได้เฉพาะ x เท่านั้น (พบ: {names})")
    return sp.sympify(expr)


def _to_func(expr: sp.Expr):
    return sp.lambdify(X, expr, modules=["numpy"])


def _evaluate(func, xs: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        ys = np.asarray(func(xs), dtype=float)
    if ys.shape == ():
        ys = np.full_like(xs, float(ys), dtype=float)
    ys[~np.isfinite(ys)] = np.nan
    return ys


def _trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.sum((ys[1:] + ys[:-1]) * np.diff(xs)) / 2)


def _critical_points(df1: sp.Expr, df2: sp.Expr, lo: float, hi: float) -> List[Dict]:
    """หาจุดวิกฤต (f'(x)=0) ในช่วง [lo, hi] และจำแนก max/min ด้วย f''"""
    points: List[Dict] = []
    try:
        roots = sp.solve(sp.Eq(df1, 0), X)
    except Exception:
        roots = []
    f_second = _to_func(df2)
    for r in roots:
        try:
            value = complex(sp.N(r))
        except Exception:
            continue
        if abs(value.imag) > 1e-9 or not (lo <= value.real <= hi):
            continue
        xr = value.real
        curvature = float(_evaluate(f_second, np.array([xr]))[0])
        if math.isnan(curvature) or abs(curvature) < 1e-9:
            kind = "ต้องตรวจสอบเพิ่ม"
        elif curvature > 0:
            kind = "ค่าต่ำสุดสัมพัทธ์ (local min)"
        else:
            kind = "ค่าสูงสุดสัมพัทธ์ (local max)"
        points.append({"x": round(xr, 6), "type": kind})
    return sorted(points, key=lambda p: p["x"])


def _critical_points_numeric(xs: np.ndarray, dys: np.ndarray) -> List[Dict]:
    """สำรอง: หาจุดที่ f'(x) เปลี่ยนเครื่องหมายบนกริด (ใช้เมื่อแก้สมการเชิงสัญลักษณ์ไม่ได้)"""
    points: List[Dict] = []
    for i in range(len(xs) - 1):
        a, b = dys[i], dys[i + 1]
        if np.isnan(a) or np.isnan(b) or a * b > 0 or a == b:
            continue
        root = xs[i] - a * (xs[i + 1] - xs[i]) / (b - a)
        if a > 0 > b:
            kind = "ค่าสูงสุดสัมพัทธ์ (local max)"
        elif a < 0 < b:
            kind = "ค่าต่ำสุดสัมพัทธ์ (local min)"
        else:
            continue
        points.append({"x": round(float(root), 4), "type": kind})
    return points


def analyze_function(expr_str: str, x0: float, x_min: float, x_max: float, n_points: int = 400) -> Dict:
    """วิเคราะห์ f(x): อนุพันธ์, เส้นสัมผัส ณ x0, จุดวิกฤต, ปริพันธ์ในช่วง [x_min, x_max]"""
    if x_min >= x_max:
        raise ValueError("x_min ต้องน้อยกว่า x_max")

    expr = parse_function(expr_str)
    d1 = sp.diff(expr, X)
    d2 = sp.diff(d1, X)

    f, f1 = _to_func(expr), _to_func(d1)
    xs = np.linspace(x_min, x_max, n_points)
    ys, dys = _evaluate(f, xs), _evaluate(f1, xs)

    f_x0 = float(_evaluate(f, np.array([x0]))[0])
    slope = float(_evaluate(f1, np.array([x0]))[0])
    if math.isnan(f_x0) or math.isnan(slope):
        raise ValueError("ฟังก์ชันไม่นิยามที่ x₀ ที่เลือก")
    tangent = f_x0 + slope * (xs - x0)
    finite = ys[~np.isnan(ys)]
    if finite.size:  # ตัดส่วนของเส้นสัมผัสที่อยู่ไกลเกินช่วงของกราฟ เพื่อไม่ให้สเกลเพี้ยน
        lo_y, hi_y = float(finite.min()), float(finite.max())
        pad = 0.25 * (hi_y - lo_y) or 1.0
        tangent = np.where((tangent < lo_y - pad) | (tangent > hi_y + pad), np.nan, tangent)

    # ปริพันธ์: ลองแบบสัญลักษณ์ก่อน แล้วเทียบกับตัวเลข (trapezoid)
    mask = ~np.isnan(ys)
    numeric_integral = _trapezoid(xs[mask], ys[mask]) if mask.sum() > 1 else float("nan")
    antiderivative_latex: Optional[str] = None
    exact_integral: Optional[float] = None
    try:
        anti = sp.integrate(expr, X)
        if not anti.has(sp.Integral):
            antiderivative_latex = sp.latex(anti) + " + C"
            exact_integral = float(sp.N(anti.subs(X, x_max) - anti.subs(X, x_min)))
    except Exception:
        pass

    curve = pd.DataFrame({"x": xs, "f(x)": ys, "f'(x)": dys, "เส้นสัมผัสที่ x₀": tangent}).set_index("x")

    return {
        "f_latex": sp.latex(expr),
        "d1_latex": sp.latex(sp.simplify(d1)),
        "d2_latex": sp.latex(sp.simplify(d2)),
        "antiderivative_latex": antiderivative_latex,
        "x0": x0,
        "f_x0": f_x0,
        "slope": slope,
        "tangent_latex": f"y = {slope:.4g}(x - {x0:.4g}) + {f_x0:.4g}",
        "numeric_integral": numeric_integral,
        "exact_integral": exact_integral,
        "critical_points": _critical_points(d1, d2, x_min, x_max) or _critical_points_numeric(xs, dys),
        "curve": curve,
    }


def gradient_descent(expr_str: str, x_start: float, learning_rate: float, n_steps: int,
                     x_min: float, x_max: float) -> Dict:
    """Gradient Descent: x ← x − η·f'(x) เพื่อหาค่าต่ำสุดของ f(x)"""
    if n_steps < 1 or learning_rate <= 0:
        raise ValueError("learning rate ต้องมากกว่า 0 และจำนวนรอบอย่างน้อย 1")
    expr = parse_function(expr_str)
    f, f1 = _to_func(expr), _to_func(sp.diff(expr, X))

    x = float(x_start)
    rows, diverged = [], False
    for step in range(n_steps + 1):
        fx = float(_evaluate(f, np.array([x]))[0])
        gx = float(_evaluate(f1, np.array([x]))[0])
        rows.append({"step": step, "x": x, "f(x)": fx, "f'(x)": gx})
        if math.isnan(gx) or math.isnan(fx) or abs(x) > 1e6:
            diverged = True
            break
        x = x - learning_rate * gx

    history = pd.DataFrame(rows)
    xs = np.linspace(x_min, x_max, 300)
    curve = pd.DataFrame({"x": xs, "f(x)": _evaluate(f, xs)}).set_index("x")
    last_grad = abs(history["f'(x)"].iloc[-1]) if not diverged else float("nan")
    return {
        "history": history,
        "curve": curve,
        "final_x": float(history["x"].iloc[-1]),
        "final_fx": float(history["f(x)"].iloc[-1]),
        "diverged": diverged,
        "converged": (not diverged) and last_grad < 1e-3,
    }


# ----------------------------------------------------- Sigmoid / Softmax
def parse_vector(text: str) -> List[float]:
    tokens = [t for t in re.split(r"[,\s;]+", (text or "").strip()) if t]
    if not tokens:
        raise ValueError("กรุณากรอกตัวเลขอย่างน้อย 1 ค่า")
    try:
        return [float(t) for t in tokens]
    except ValueError:
        raise ValueError("ข้อมูลต้องเป็นตัวเลขคั่นด้วยเครื่องหมายจุลภาค (,) หรือช่องว่าง") from None


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Softmax แบบเสถียรเชิงตัวเลข (ลบค่าสูงสุดก่อน) พร้อม temperature"""
    if temperature <= 0:
        raise ValueError("temperature ต้องมากกว่า 0")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def activation_report(logits_text: str, temperature: float, labels_text: str = "") -> Dict:
    logits = parse_vector(logits_text)
    n = len(logits)
    labels = [s.strip() for s in labels_text.split(",") if s.strip()] if labels_text else []
    if len(labels) != n:
        labels = [f"คลาส {i + 1}" for i in range(n)]

    probs = softmax(logits, temperature)
    base = softmax(logits, 1.0)
    entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1))))

    table = pd.DataFrame({
        "คลาส": labels, "logit": logits,
        "sigmoid(logit)": np.round(sigmoid(logits), 4),
        "softmax (T=1)": np.round(base, 4),
        f"softmax (T={temperature:g})": np.round(probs, 4),
    })

    zs = np.linspace(-8, 8, 161)
    s = sigmoid(zs)
    curve = pd.DataFrame({"z": zs, "sigmoid(z)": s, "sigmoid'(z) = σ(1−σ)": s * (1 - s)}).set_index("z")

    temps = [0.25, 0.5, 1.0, 2.0, 5.0]
    sweep = pd.DataFrame({f"T={t:g}": softmax(logits, t) for t in temps}, index=labels)

    return {
        "table": table, "curve": curve, "temperature_sweep": sweep,
        "probabilities": pd.DataFrame({"ความน่าจะเป็น": probs}, index=labels),
        "entropy": entropy, "max_entropy": math.log2(n) if n > 1 else 0.0,
        "top_label": labels[int(np.argmax(probs))], "top_prob": float(probs.max()),
        "sum_check": float(probs.sum()),
    }


# ------------------------------------------------------- Linear Algebra
def parse_matrix(text: str) -> np.ndarray:
    """แต่ละบรรทัด = 1 แถว, ค่าในแถวคั่นด้วย , หรือช่องว่าง"""
    rows = [r for r in (text or "").strip().splitlines() if r.strip()]
    if not rows:
        raise ValueError("กรุณากรอกเมทริกซ์")
    try:
        data = [[float(t) for t in re.split(r"[,\s;]+", r.strip()) if t] for r in rows]
    except ValueError:
        raise ValueError("เมทริกซ์ต้องประกอบด้วยตัวเลขเท่านั้น") from None
    if len({len(r) for r in data}) != 1:
        raise ValueError("ทุกแถวต้องมีจำนวนคอลัมน์เท่ากัน")
    return np.array(data, dtype=float)


def _fmt_complex(c: complex) -> str:
    c = complex(c)
    if abs(c.imag) < 1e-10:
        return f"{c.real:.4f}"
    return f"{c.real:.4f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.4f}i"


def transform_shape(A: np.ndarray, points: int = 120) -> pd.DataFrame:
    """ผลของการแปลงเชิงเส้น A (2×2) ต่อวงกลมหนึ่งหน่วย → กลายเป็นวงรี"""
    t = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack([np.cos(t), np.sin(t)])
    moved = A @ circle
    return pd.DataFrame({
        "x": np.concatenate([circle[0], moved[0]]),
        "y": np.concatenate([circle[1], moved[1]]),
        "รูปทรง": ["วงกลมเดิม"] * points + ["หลังแปลงด้วย A"] * points,
    })


def analyze_matrix(a_text: str, b_text: str = "") -> Dict:
    A = parse_matrix(a_text)
    m, n = A.shape
    result: Dict = {
        "shape": (m, n), "rank": int(np.linalg.matrix_rank(A)),
        "matrix": pd.DataFrame(A), "is_square": m == n,
        "transpose": pd.DataFrame(A.T),
    }
    if m == n:
        det = float(np.linalg.det(A))
        result.update(trace=float(np.trace(A)), determinant=det, invertible=abs(det) > 1e-10)
        if result["invertible"]:
            result["inverse"] = pd.DataFrame(np.linalg.inv(A))
            result["condition_number"] = float(np.linalg.cond(A))
        values, vectors = np.linalg.eig(A)
        result["eigen"] = [
            {"eigenvalue": _fmt_complex(values[i]),
             "eigenvector": "[" + ", ".join(_fmt_complex(v) for v in vectors[:, i]) + "]"}
            for i in range(n)
        ]
        if n == 2:
            result["transform"] = transform_shape(A)
    if b_text and b_text.strip():
        b = np.array(parse_vector(b_text), dtype=float)
        if len(b) != m:
            raise ValueError(f"เวกเตอร์ b ต้องมี {m} ค่า (เท่าจำนวนแถวของ A)")
        if m == n and result.get("invertible"):
            result["solution"] = np.linalg.solve(A, b).tolist()
            result["solution_type"] = "ผลเฉลยเดียว (Ax = b)"
        else:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            result["solution"] = sol.tolist()
            result["solution_type"] = "ผลเฉลยกำลังสองน้อยสุด (Least Squares)"
        result["residual"] = float(np.linalg.norm(A @ np.array(result["solution"]) - b))
    return result


def vector_report(u_text: str, v_text: str) -> Dict:
    u, v = np.array(parse_vector(u_text)), np.array(parse_vector(v_text))
    if len(u) != len(v):
        raise ValueError("เวกเตอร์ u และ v ต้องมีจำนวนมิติเท่ากัน")
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    dot = float(u @ v)
    out = {"dot": dot, "norm_u": nu, "norm_v": nv, "distance": float(np.linalg.norm(u - v)),
           "cosine": None, "angle_deg": None, "projection": None, "cross": None}
    if nu > 0 and nv > 0:
        cos = max(-1.0, min(1.0, dot / (nu * nv)))
        out["cosine"], out["angle_deg"] = cos, math.degrees(math.acos(cos))
    if nv > 0:
        out["projection"] = (dot / nv ** 2 * v).tolist()
    if len(u) == 3:
        out["cross"] = np.cross(u, v).tolist()
    return out
