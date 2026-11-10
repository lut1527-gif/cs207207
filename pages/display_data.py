"""หน้า UI: แสดงข้อมูลดิบจากโฟลเดอร์ data/"""
import json

import streamlit as st

from services import data_loader as dl

st.title("🗂️ ข้อมูลดิบ (Raw Data)")
st.caption("ข้อมูลที่อ่านจากโฟลเดอร์ `data/` ซึ่งใช้ร่วมกันในหน้า Statistics และ Markov")

try:
    raw = dl.load_raw_data()
except (FileNotFoundError, json.JSONDecodeError) as err:
    st.error(f"อ่านไฟล์ข้อมูลไม่สำเร็จ: {err}")
    st.stop()

files = dl.list_data_files()
m1, m2, m3 = st.columns(3)
m1.metric("ไฟล์ใน data/", len(files))
m2.metric("ชุดข้อความ", len(raw.get("text_datasets", [])))
m3.metric("ชุดตัวเลข", len(raw.get("number_sets", [])))

tab_text, tab_num, tab_json, tab_files = st.tabs(["📄 ชุดข้อความ", "🔢 ชุดตัวเลข", "{ } JSON ดิบ", "📁 ไฟล์ทั้งหมด"])

with tab_text:
    st.dataframe(dl.text_datasets_table())
    for d in dl.get_text_datasets():
        with st.expander(f"{d['title']}  ({d['id']})"):
            st.write(d["text"])

with tab_num:
    st.dataframe(dl.number_sets_table())

with tab_json:
    st.json(raw, expanded=1)
    st.download_button(
        "⬇️ ดาวน์โหลด raw_data.json",
        data=json.dumps(raw, ensure_ascii=False, indent=2),
        file_name="raw_data.json",
        mime="application/json",
    )

with tab_files:
    st.dataframe(files)
