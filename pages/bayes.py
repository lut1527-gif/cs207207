"""หน้า UI: Bayes' Theorem Calculator (ไม่มีสูตรคำนวณในไฟล์นี้)"""
import streamlit as st

from services import bayes_service as bayes

st.title("🎲 Bayes' Theorem Calculator")
st.latex(r"P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}")

PRESETS = {
    "กำหนดเอง": (0.01, 0.99, 0.05),
    "ตรวจโรค (ความชุก 1%, ตรวจเจอจริง 99%, ผลบวกลวง 5%)": (0.01, 0.99, 0.05),
    "กรองสแปม (สแปม 40%, คำว่า 'ฟรี' ในสแปม 60%, ในเมลปกติ 5%)": (0.40, 0.60, 0.05),
}


def _apply_preset() -> None:
    prior, lik, lik_not = PRESETS[st.session_state["bayes_preset"]]
    st.session_state.update(
        bayes_prior=prior,
        bayes_lik=lik,
        bayes_lik_not=lik_not,
        bayes_evidence=bayes.calculate_evidence(prior, lik, lik_not),
    )


for key, default in (("bayes_prior", 0.01), ("bayes_lik", 0.99), ("bayes_lik_not", 0.05)):
    st.session_state.setdefault(key, default)

st.selectbox("ตัวอย่างโจทย์", list(PRESETS), key="bayes_preset", on_change=_apply_preset)
mode = st.radio(
    "วิธีระบุ Evidence P(B)",
    ["กรอก P(B) โดยตรง", "คำนวณ P(B) จาก P(B|¬A) (Law of Total Probability)"],
    horizontal=True,
)
direct = mode.startswith("กรอก")

c1, c2, c3, c4 = st.columns(4)
prior = c1.number_input("Prior  P(A)", 0.0, 1.0, step=0.01, format="%.4f", key="bayes_prior")
likelihood = c2.number_input("Likelihood  P(B|A)", 0.0, 1.0, step=0.01, format="%.4f", key="bayes_lik")
likelihood_not = c3.number_input(
    "P(B|¬A)", 0.0, 1.0, step=0.01, format="%.4f", key="bayes_lik_not",
    disabled=direct, help="โอกาสเจอ B ทั้งที่ A ไม่จริง (ใช้ในโหมดคำนวณ P(B))",
)
if direct:
    if "bayes_evidence" not in st.session_state:
        st.session_state["bayes_evidence"] = bayes.calculate_evidence(prior, likelihood, likelihood_not)
    evidence = c4.number_input("Evidence  P(B)", 0.0, 1.0, step=0.01, format="%.4f", key="bayes_evidence")
    likelihood_not = None
else:
    evidence = bayes.calculate_evidence(prior, likelihood, likelihood_not)
    c4.metric("Evidence  P(B)", f"{evidence:.4f}", help="P(B|A)P(A) + P(B|¬A)P(¬A)")

try:
    report = bayes.bayes_report(prior, likelihood, evidence)
except ValueError as err:
    st.error(f"⚠️ {err}")
    st.stop()

st.subheader("ผลลัพธ์")
m1, m2, m3 = st.columns(3)
m1.metric("Posterior  P(A|B)", f"{report['posterior']:.4%}", delta=f"{report['posterior'] - prior:+.2%} จาก Prior")
m2.metric("P(¬A|B)", f"{report['posterior_not']:.4%}")
m3.metric("Evidence  P(B)", f"{report['evidence']:.4f}")

st.latex(
    rf"P(A\mid B)=\frac{{{likelihood:.4f}\times{prior:.4f}}}{{{evidence:.4f}}}={report['posterior']:.4f}"
)
st.success(report["summary"])

left, right = st.columns(2)
with left:
    st.markdown("**เปรียบเทียบ Prior กับ Posterior**")
    st.bar_chart({"Prior P(A)": [prior], "Posterior P(A|B)": [report["posterior"]]})
with right:
    if likelihood_not is not None:
        st.markdown("**Posterior จะเปลี่ยนไปอย่างไรเมื่อ Prior เปลี่ยน**")
        st.line_chart(bayes.sensitivity_curve(likelihood, likelihood_not), x="prior", y="posterior")
    else:
        st.info("เลือกโหมด 'คำนวณ P(B) จาก P(B|¬A)' เพื่อดูกราฟ Sensitivity และตารางจำนวนคน")

if likelihood_not is not None:
    with st.expander("🧑‍🤝‍🧑 อธิบายด้วยจำนวนคน (สมมติ 10,000 คน)", expanded=True):
        st.dataframe(bayes.natural_frequencies(prior, likelihood, likelihood_not))
        st.caption("Posterior = (A จริงและพบ B) ÷ (ทุกคนที่พบ B)")
