"""หน้า UI: ML Math Lab — แอปประยุกต์ (Calculus · Sigmoid/Softmax · Linear Algebra)
ทุกการคำนวณอยู่ใน services/custom_service.py"""
import streamlit as st

from services import custom_service as lab

st.title("🧪 ML Math Lab")
st.caption("ห้องทดลองคณิตศาสตร์สำหรับ Machine Learning — เปลี่ยนค่าแล้วเห็นผลทันที")

tab_calc, tab_gd, tab_act, tab_mat, tab_vec = st.tabs(
    ["📐 Calculus", "⛰️ Gradient Descent", "📉 Sigmoid / Softmax", "🔢 Matrix", "➡️ Vector"]
)

# ---------------------------------------------------------------- Calculus
with tab_calc:
    st.markdown("พิมพ์ฟังก์ชันของ `x` เช่น `x^3 - 3x`, `sin(x)`, `exp(-x^2)`, `sigmoid(x)`")
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    expr = c1.text_input("f(x) =", "x^3 - 3x", key="calc_expr")
    x_min = c2.number_input("x เริ่มต้น", value=-3.0, step=0.5, key="calc_xmin")
    x_max = c3.number_input("x สิ้นสุด", value=3.0, step=0.5, key="calc_xmax")
    x0 = c4.number_input("จุดสัมผัส x₀", value=2.0, step=0.5, key="calc_x0")
    try:
        res = lab.analyze_function(expr, x0, x_min, x_max)
    except ValueError as err:
        st.error(f"⚠️ {err}")
    else:
        l, r = st.columns(2)
        with l:
            st.latex(rf"f(x)={res['f_latex']}")
            st.latex(rf"f'(x)={res['d1_latex']}")
            st.latex(rf"f''(x)={res['d2_latex']}")
            if res["antiderivative_latex"]:
                st.latex(rf"\int f(x)\,dx={res['antiderivative_latex']}")
        with r:
            k1, k2, k3 = st.columns(3)
            k1.metric(f"f({x0:g})", f"{res['f_x0']:.4f}")
            k2.metric(f"f'({x0:g}) ความชัน", f"{res['slope']:.4f}")
            k3.metric(f"∫ จาก {x_min:g} ถึง {x_max:g}", f"{res['numeric_integral']:.4f}")
            st.caption(f"เส้นสัมผัส: {res['tangent_latex']}")
            if res["exact_integral"] is not None:
                st.caption(f"ค่าปริพันธ์แบบแม่นตรง = {res['exact_integral']:.4f} (เทียบกับวิธี Trapezoid ด้านบน)")
        st.markdown("**กราฟ f(x) และเส้นสัมผัส ณ จุด x₀**")
        st.line_chart(res["curve"][["f(x)", "เส้นสัมผัสที่ x₀"]])
        st.markdown("**กราฟอนุพันธ์ f′(x)** — จุดที่กราฟตัดแกน 0 คือจุดวิกฤตของ f(x)")
        st.line_chart(res["curve"][["f'(x)"]])
        if res["critical_points"]:
            st.markdown("**จุดวิกฤตในช่วงที่เลือก**")
            st.dataframe(res["critical_points"], column_config={"x": "ตำแหน่ง x", "type": "ชนิด"})
        else:
            st.info("ไม่พบจุดวิกฤตในช่วงที่เลือก")

# ---------------------------------------------------------- Gradient Descent
with tab_gd:
    st.latex(r"x_{t+1}=x_t-\eta\,f'(x_t)")
    g1, g2, g3, g4 = st.columns([3, 1, 1, 1])
    gexpr = g1.text_input("f(x) ที่ต้องการหาค่าต่ำสุด", "(x-3)^2 + 1", key="gd_expr")
    x_start = g2.number_input("จุดเริ่มต้น x₀", value=10.0, step=1.0, key="gd_x0")
    lr = g3.number_input("Learning rate η", min_value=0.001, max_value=5.0, value=0.1, step=0.05, format="%.3f", key="gd_lr")
    steps = g4.number_input("จำนวนรอบ", min_value=1, max_value=500, value=30, step=5, key="gd_steps")
    try:
        gd = lab.gradient_descent(gexpr, x_start, lr, int(steps), min(x_start, 0.0) - 5, max(x_start, 0.0) + 5)
    except ValueError as err:
        st.error(f"⚠️ {err}")
    else:
        if gd["diverged"]:
            st.error("❌ ค่าแตกออก (diverge) — ลองลด learning rate")
        elif gd["converged"]:
            st.success("✅ ลู่เข้าสู่ค่าต่ำสุดแล้ว (|f′(x)| < 0.001)")
        else:
            st.warning("⏳ ยังไม่ลู่เข้า — ลองเพิ่มจำนวนรอบหรือปรับ learning rate")
        k1, k2 = st.columns(2)
        k1.metric("x สุดท้าย", f"{gd['final_x']:.5f}")
        k2.metric("f(x) สุดท้าย", f"{gd['final_fx']:.5f}")
        p1, p2 = st.columns(2)
        p1.markdown("**ค่า f(x) ในแต่ละรอบ**")
        p1.line_chart(gd["history"], x="step", y="f(x)")
        p2.markdown("**กราฟ f(x)**")
        p2.line_chart(gd["curve"])
        with st.expander("ตารางประวัติการปรับค่า"):
            st.dataframe(gd["history"])

# ---------------------------------------------------------- Sigmoid / Softmax
with tab_act:
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}\qquad \text{softmax}(z_i)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}")
    a1, a2 = st.columns(2)
    logits = a1.text_input("ค่า logits (คั่นด้วย ,)", "2.0, 1.0, 0.1, -1.0", key="act_logits")
    labels = a2.text_input("ชื่อคลาส (ไม่บังคับ)", "แมว, สุนัข, นก, ปลา", key="act_labels")
    temp = st.slider("Temperature (T)", 0.1, 5.0, 1.0, 0.1, key="act_temp")
    try:
        act = lab.activation_report(logits, temp, labels)
    except ValueError as err:
        st.error(f"⚠️ {err}")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("คลาสที่ทำนาย", act["top_label"], f"{act['top_prob']:.1%}")
        k2.metric("Entropy (bits)", f"{act['entropy']:.3f}", help=f"ค่าสูงสุดที่เป็นไปได้ = {act['max_entropy']:.3f}")
        k3.metric("ผลรวมความน่าจะเป็น", f"{act['sum_check']:.4f}")
        st.dataframe(act["table"])
        l, r = st.columns(2)
        l.markdown("**ความน่าจะเป็นจาก Softmax**")
        l.bar_chart(act["probabilities"])
        r.markdown("**Sigmoid และอนุพันธ์**")
        r.line_chart(act["curve"])
        st.markdown("**Temperature ส่งผลอย่างไร?** — T ต่ำ ⇒ มั่นใจในคำตอบเดียว, T สูง ⇒ ความน่าจะเป็นกระจายเท่าๆ กัน")
        st.dataframe(act["temperature_sweep"])

# -------------------------------------------------------------------- Matrix
with tab_mat:
    m1, m2 = st.columns([2, 1])
    a_text = m1.text_area("เมทริกซ์ A (1 แถวต่อ 1 บรรทัด)", "2 1\n1 3", height=120, key="mat_a")
    b_text = m2.text_input("เวกเตอร์ b สำหรับแก้ Ax = b (ไม่บังคับ)", "5, 10", key="mat_b")
    try:
        mat = lab.analyze_matrix(a_text, b_text)
    except Exception as err:  # noqa: BLE001 — ValueError จาก service หรือ LinAlgError จาก numpy
        st.error(f"⚠️ {err}")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ขนาด", f"{mat['shape'][0]}×{mat['shape'][1]}")
        k2.metric("Rank", mat["rank"])
        if mat["is_square"]:
            k3.metric("Determinant", f"{mat['determinant']:.4f}")
            k4.metric("Trace", f"{mat['trace']:.4f}")
        l, r = st.columns(2)
        l.markdown("**Transpose Aᵀ**")
        l.dataframe(mat["transpose"])
        if mat["is_square"]:
            if mat["invertible"]:
                r.markdown(f"**Inverse A⁻¹** (condition number = {mat['condition_number']:.2f})")
                r.dataframe(mat["inverse"])
            else:
                r.warning("det(A) = 0 ⇒ เมทริกซ์ไม่มีอินเวอร์ส (singular)")
            st.markdown("**Eigenvalues / Eigenvectors**")
            st.dataframe(mat["eigen"], column_config={"eigenvalue": "Eigenvalue (λ)", "eigenvector": "Eigenvector (v)"})
        else:
            st.info("Determinant, Inverse และ Eigen ใช้ได้เฉพาะเมทริกซ์จัตุรัส")
        if "solution" in mat:
            st.markdown(f"**{mat['solution_type']}**")
            st.write({f"x{i + 1}": round(v, 6) for i, v in enumerate(mat["solution"])})
            st.caption(f"ค่าคลาดเคลื่อน ‖Ax − b‖ = {mat['residual']:.2e}")
        if "transform" in mat:
            st.markdown("**การแปลงเชิงเส้น: A ยืด/บิดวงกลมหนึ่งหน่วยให้กลายเป็นวงรี** (แกนของวงรีสัมพันธ์กับ Eigenvector)")
            st.scatter_chart(mat["transform"], x="x", y="y", color="รูปทรง")

# -------------------------------------------------------------------- Vector
with tab_vec:
    v1, v2 = st.columns(2)
    u_text = v1.text_input("เวกเตอร์ u", "1, 2, 3", key="vec_u")
    v_text = v2.text_input("เวกเตอร์ v", "4, 5, 6", key="vec_v")
    try:
        vec = lab.vector_report(u_text, v_text)
    except ValueError as err:
        st.error(f"⚠️ {err}")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("u · v (Dot product)", f"{vec['dot']:.4f}")
        k2.metric("‖u‖", f"{vec['norm_u']:.4f}")
        k3.metric("‖v‖", f"{vec['norm_v']:.4f}")
        k4.metric("ระยะห่าง ‖u − v‖", f"{vec['distance']:.4f}")
        if vec["cosine"] is not None:
            st.metric("Cosine similarity / มุมระหว่างเวกเตอร์", f"{vec['cosine']:.4f}", f"{vec['angle_deg']:.2f}°", delta_color="off")
        if vec["projection"] is not None:
            st.write("**Projection ของ u บน v:**", [round(x, 4) for x in vec["projection"]])
        if vec["cross"] is not None:
            st.write("**Cross product u × v:**", [round(x, 4) for x in vec["cross"]])
