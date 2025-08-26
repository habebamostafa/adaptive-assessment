# app.py
import streamlit as st
import requests

st.title("🎓 نظام الطلاب - n8n فقط")

N8N_URL = "http://localhost:5678"

tab1, tab2 = st.tabs(["تسجيل جديد", "تسجيل الدخول"])

with tab1:
    st.header("تسجيل طالب جديد")
    with st.form("register"):
        name = st.text_input("الاسم")
        email = st.text_input("البريد الإلكتروني")
        password = st.text_input("كلمة المرور", type="password")
        track = st.selectbox("التخصص", ["AI", "Web", "Data"])
        
        if st.form_submit_button("تسجيل"):
            response = requests.post(f"{N8N_URL}/register", json={
                "name": name, "email": email, "password": password, "track": track
            })
            st.write(response.json())

with tab2:
    st.header("تسجيل الدخول")
    with st.form("login"):
        email = st.text_input("البريد الإلكتروني")
        password = st.text_input("كلمة المرور", type="password")
        
        if st.form_submit_button("دخول"):
            response = requests.post(f"{N8N_URL}/login", json={
                "email": email, "password": password
            })
            st.write(response.json())