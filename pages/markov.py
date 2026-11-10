"""หน้า UI: ทำนายคำถัดไปด้วย Markov Chain (logic อยู่ใน services/text_prediction.py)"""
import streamlit as st

from services import text_prediction as tp
from services.data_loader import get_text_datasets

st.title("🔗 Markov Chain — ทำนายคำถัดไป")
st.latex(r"P(w_{t+1}\mid w_t)=\frac{\text{count}(w_t,\,w_{t+1})}{\text{count}(w_t)}")

datasets = {d["title"]: d["text"] for d in get_text_datasets()}
CUSTOM = "✍️ กรอกข้อความเอง"

col_a, col_b = st.columns([3, 1])
choice = col_a.selectbox("เลือกชุดข้อมูลอ้างอิง (จาก data/raw_data.json)", [*datasets, CUSTOM])
order = col_b.slider("Order (จำนวนคำใน state)", 1, 3, 1)

if choice == CUSTOM:
    source_text = st.text_area("ข้อความอ้างอิงของคุณ", height=140,
                               value="the cat sat on the mat and the cat saw the dog on the mat")
else:
    source_text = datasets[choice]
    with st.expander("ดูข้อความต้นฉบับ"):
        st.write(source_text)

try:
    model = tp.build_model(source_text, order)
except ValueError as err:
    st.error(f"⚠️ {err}")
    st.stop()

info = tp.model_summary(model)
i1, i2, i3 = st.columns(3)
i1.metric("จำนวนคำทั้งหมด", info["total_words"])
i2.metric("คำศัพท์ไม่ซ้ำ", info["vocabulary_size"])
i3.metric("จำนวน State", info["n_states"])

tab_next, tab_gen, tab_matrix = st.tabs(["🔮 ทำนายคำถัดไป", "📝 สร้างประโยค", "🧮 Transition Matrix"])

with tab_next:
    context = st.text_input("ข้อความอ้างอิงเริ่มต้น", value="the", key="mk_context")
    top_k = st.slider("แสดงกี่อันดับ", 1, 10, 5)
    try:
        prediction = tp.predict_next(model, context, top_k)
    except ValueError as err:
        st.warning(str(err))
    else:
        best = prediction["predictions"][0]
        st.success(f"คำถัดไปที่น่าจะเป็นที่สุดจาก state «{prediction['state']}» คือ **{best['word']}** ({best['probability']:.1%})")
        if prediction["used_backoff"]:
            st.caption("ℹ️ ไม่พบ state ตรงตัว จึงใช้ state ที่ลงท้ายด้วยคำสุดท้ายของข้อความแทน")
        st.dataframe(
            prediction["predictions"],
            column_config={
                "word": "คำ",
                "count": "จำนวนครั้งที่พบ",
                "probability": st.column_config.ProgressColumn(
                    "ความน่าจะเป็น", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

with tab_gen:
    g1, g2 = st.columns(2)
    length = g1.slider("จำนวนคำที่ต้องการสร้างต่อ", 5, 60, 20)
    seed = g2.number_input("Random seed (เปลี่ยนเพื่อให้ได้ประโยคใหม่)", 0, 9999, 42)
    try:
        st.markdown("**ผลลัพธ์:**")
        st.info(tp.generate_text(model, st.session_state["mk_context"], length, int(seed)))
    except ValueError as err:
        st.warning(str(err))

with tab_matrix:
    st.caption("แถว = state ปัจจุบัน, คอลัมน์ = คำถัดไป (แสดงเฉพาะ state/คำที่พบบ่อยที่สุด) — แต่ละแถวรวมกันได้ไม่เกิน 1 เพราะตัดคำที่พบน้อยออก")
    st.dataframe(tp.transition_matrix(model))
