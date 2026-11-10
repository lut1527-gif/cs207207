"""หน้า UI: สถิติเชิงพรรณนา (ไม่มีสูตรคำนวณในไฟล์นี้ — เรียกจาก services)"""
import streamlit as st

from services import cal_statistics as stats
from services.data_loader import get_number_sets

st.title("📈 สถิติเชิงพรรณนา")
st.caption("กรอกชุดตัวเลข แล้วระบบจะคำนวณ Mean, Median, Mode, Standard Deviation และ Variance ให้ทันที")

example_sets = {s["title"]: s["values"] for s in get_number_sets()}


def _load_example() -> None:
    values = example_sets[st.session_state["stat_example"]]
    st.session_state["stat_text"] = ", ".join(str(v) for v in values)


if "stat_text" not in st.session_state:
    st.session_state["stat_text"] = "-100, 0, 10, 20, 30, 100"

col_in, col_opt = st.columns([3, 2])
with col_in:
    st.text_area(
        "ชุดตัวเลข (คั่นด้วยเครื่องหมายจุลภาค ช่องว่าง หรือขึ้นบรรทัดใหม่)",
        key="stat_text",
        height=120,
    )
with col_opt:
    st.selectbox("โหลดชุดข้อมูลตัวอย่าง", list(example_sets), key="stat_example", on_change=_load_example)
    kind = st.radio(
        "ประเภทของ SD / Variance",
        ["ประชากร (หาร N)", "ตัวอย่าง (หาร N−1)"],
        horizontal=True,
    )

try:
    values = stats.parse_numbers(st.session_state["stat_text"])
    result = stats.summarize(values, sample=kind.startswith("ตัวอย่าง"))
except ValueError as err:
    st.error(f"⚠️ {err}")
    st.stop()

st.subheader("ผลลัพธ์")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Mean (ค่าเฉลี่ย)", f"{result['mean']:.4f}")
m2.metric("Median (มัธยฐาน)", f"{result['median']:.4f}")
mode_info = result["mode"]
m3.metric("Mode (ฐานนิยม)", ", ".join(f"{v:g}" for v in mode_info["modes"]) if mode_info["has_mode"] else "ไม่มี")
m4.metric("SD", f"{result['sd']:.4f}")
m5.metric("Variance", f"{result['variance']:.4f}")

n1, n2, n3, n4, n5 = st.columns(5)
n1.metric("จำนวนข้อมูล (N)", result["count"])
n2.metric("ค่าต่ำสุด", f"{result['min']:g}")
n3.metric("ค่าสูงสุด", f"{result['max']:g}")
n4.metric("พิสัย (Range)", f"{result['range']:g}")
n5.metric("IQR", f"{result['iqr']:.4f}")

if mode_info["has_mode"] and len(mode_info["modes"]) > 1:
    st.info(f"ข้อมูลนี้มีหลายฐานนิยม (multimodal) — แต่ละค่าปรากฏ {mode_info['frequency']} ครั้ง")
elif not mode_info["has_mode"]:
    st.info("ทุกค่าปรากฏเพียงครั้งเดียว จึงไม่มีฐานนิยม")

tab_chart, tab_table, tab_formula = st.tabs(["📊 ฮิสโตแกรม", "📋 ตารางข้อมูล", "🧮 สูตรที่ใช้"])
with tab_chart:
    bins = st.slider("จำนวนช่วง (bins)", 2, 20, min(6, max(2, len(set(values)))))
    st.bar_chart(stats.histogram_data(values, bins), x="midpoint", y="count")
with tab_table:
    left, right = st.columns(2)
    left.markdown("**ข้อมูลที่เรียงลำดับแล้ว**")
    left.write(result["sorted"])
    right.markdown("**ตารางความถี่**")
    right.dataframe(result["frequency"])
with tab_formula:
    st.latex(r"\bar{x}=\frac{1}{N}\sum_{i=1}^{N}x_i")
    st.latex(r"\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\bar{x})^2 \quad,\quad s^2=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\bar{x})^2")
    st.latex(r"SD=\sqrt{\text{Variance}}")
    st.markdown("**Median** = ค่ากลางของข้อมูลที่เรียงลำดับแล้ว (ถ้า N เป็นเลขคู่ ใช้ค่าเฉลี่ยของสองค่ากลาง) · **Mode** = ค่าที่ปรากฏบ่อยที่สุด")
