# secA-1-project — Probability & Applied Math Lab

Streamlit mono-repo: UI (`pages/`) แยกจาก Business Logic (`services/`)

## วิธีรัน
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```
เปิดที่ http://localhost:8501

## ก่อนส่งงาน ต้องแก้
1. `data/members.json` — ใส่ชื่อ-นามสกุล รหัสนักศึกษา บทบาท และหน้าที่จริงของสมาชิกทุกคน
2. `data/imgs/` — แทนที่ `student-a.jpeg` ... ด้วยรูปถ่ายจริง (ชื่อไฟล์ต้องตรงกับ `image` ใน members.json)
3. เปลี่ยนชื่อโฟลเดอร์ให้ตรงกลุ่มของคุณ: `sec{section}-{group}-project`

## โครงสร้าง
| ไฟล์ | หน้าที่ |
|---|---|
| `main.py` | Routing (`st.navigation`) |
| `pages/home.py` | หน้าแรก + เมนู |
| `pages/statistics.py` ↔ `services/cal_statistics.py`, `cal_mode.py` | Mean/Median/Mode/SD/Variance |
| `pages/bayes.py` ↔ `services/bayes_service.py` | Bayes' Theorem |
| `pages/markov.py` ↔ `services/text_prediction.py` | Markov Chain |
| `pages/display_data.py` ↔ `services/data_loader.py` | แสดงข้อมูลดิบ |
| `pages/custom_app.py` ↔ `services/custom_service.py` | ML Math Lab |
| `pages/members.py` | สมาชิกกลุ่ม |
