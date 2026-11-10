"""Main entry point / Routing logic — รันด้วย:  streamlit run main.py"""
import streamlit as st

st.set_page_config(
    page_title="Probability & Applied Math Lab",
    page_icon="📊",
    layout="wide",
)

pages = [
    st.Page("pages/home.py", title="หน้าแรก", icon="🏠", default=True),
    st.Page("pages/statistics.py", title="สถิติเชิงพรรณนา", icon="📈"),
    st.Page("pages/bayes.py", title="Bayes' Theorem", icon="🎲"),
    st.Page("pages/markov.py", title="Markov Chain", icon="🔗"),
    st.Page("pages/display_data.py", title="ข้อมูลดิบ", icon="🗂️"),
    st.Page("pages/custom_app.py", title="ML Math Lab", icon="🧪"),
    st.Page("pages/members.py", title="สมาชิกกลุ่ม", icon="👥"),
]

navigation = st.navigation(pages)
st.sidebar.caption("secA-1-project · Streamlit Mono-repo")
navigation.run()
