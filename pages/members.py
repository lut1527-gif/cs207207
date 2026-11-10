"""หน้า UI: สมาชิกกลุ่ม (อ่านข้อมูลจาก data/members.json + data/imgs/)"""
import streamlit as st

from services.data_loader import load_members

st.title("👥 สมาชิกกลุ่ม")

try:
    info = load_members()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

st.caption(f"Section {info['section']} · Group {info['group']} · โปรเจกต์ `{info['project_name']}`")

members = info["members"]
PER_ROW = 3
for start in range(0, len(members), PER_ROW):
    cols = st.columns(PER_ROW)
    for col, m in zip(cols, members[start:start + PER_ROW]):
        with col.container(border=True):
            if m["image_path"]:
                st.image(m["image_path"], width=200)
            else:
                st.warning(f"ไม่พบรูป {m['image']}")
            st.markdown(f"### {m['name']}")
            st.markdown(f"**รหัสนักศึกษา:** {m['student_id']}")
            st.markdown(f"**บทบาท:** {m['role']}")
            st.markdown("**หน้าที่ความรับผิดชอบ:**")
            st.markdown("\n".join(f"- {r}" for r in m["responsibilities"]))

st.subheader("สรุปการแบ่งงาน")
st.dataframe(
    [{"ชื่อ-นามสกุล": m["name"], "รหัสนักศึกษา": m["student_id"], "บทบาท": m["role"],
      "หน้าที่": " / ".join(m["responsibilities"])} for m in members]
)
