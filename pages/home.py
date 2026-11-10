"""Landing page: ภาพรวมระบบ + ปุ่มนำทางไปแอปย่อย (UI เท่านั้น)"""
import streamlit as st

st.title("📊 Probability & Applied Math Lab")
st.markdown(
    "เว็บแอปพลิเคชันสำหรับ **คำนวณสถิติ ความน่าจะเป็น Markov Chain และคณิตศาสตร์ประยุกต์** "
    "พัฒนาด้วย Python + Streamlit แยกส่วน UI (`pages/`) ออกจาก Business Logic (`services/`) อย่างชัดเจน"
)

apps = [
    ("📈", "สถิติเชิงพรรณนา", "pages/statistics.py",
     "Mean, Median, Mode, SD, Variance พร้อมฮิสโตแกรมและตารางความถี่"),
    ("🎲", "Bayes' Theorem", "pages/bayes.py",
     "คำนวณ Posterior จาก Prior, Likelihood, Evidence พร้อมกราฟอธิบาย"),
    ("🔗", "Markov Chain", "pages/markov.py",
     "ทำนายคำถัดไป สร้างประโยค และดู Transition Matrix"),
    ("🗂️", "ข้อมูลดิบ", "pages/display_data.py",
     "ดูข้อมูลทั้งหมดในโฟลเดอร์ data/ ในรูปแบบตารางและ JSON"),
    ("🧪", "ML Math Lab (Custom App)", "pages/custom_app.py",
     "อนุพันธ์/ปริพันธ์, Gradient Descent, Sigmoid/Softmax, เมทริกซ์และเวกเตอร์"),
    ("👥", "สมาชิกกลุ่ม", "pages/members.py",
     "รายชื่อ รหัสนักศึกษา รูปถ่าย และหน้าที่ความรับผิดชอบ"),
]

st.subheader("เมนูหลัก")
for row_start in range(0, len(apps), 3):
    cols = st.columns(3)
    for col, (icon, title, path, desc) in zip(cols, apps[row_start:row_start + 3]):
        with col.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(desc)
            st.page_link(path, label=f"เปิด {title}", icon="➡️")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("สถาปัตยกรรมระบบ")
    st.markdown(
        "- **`main.py`** — Routing / จุดเริ่มต้นของแอป\n"
        "- **`pages/`** — UI: รับค่าและแสดงผลเท่านั้น\n"
        "- **`services/`** — Logic: คำนวณแล้วคืนค่ากลับไปให้ UI\n"
        "- **`data/`** — ข้อมูลดิบ (`raw_data.json`, `members.json`) และรูปภาพ"
    )
with right:
    st.subheader("หัวข้อคณิตศาสตร์ที่ใช้")
    st.table({
        "หัวข้อ": ["สถิติเชิงพรรณนา", "Bayes' Theorem", "Markov Chain", "Calculus", "Linear Algebra"],
        "ใช้ในหน้า": ["สถิติ", "Bayes", "Markov", "ML Math Lab", "ML Math Lab"],
    })
