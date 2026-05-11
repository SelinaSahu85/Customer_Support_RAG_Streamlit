import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.title("Customer Support Assistant")

user_input = st.text_input("Message")

if st.button("Send"):
    if user_input:
        st.write("🟢 You:", user_input)

        res = requests.post(API_URL, json={"query": user_input})

        if res.status_code == 200:
            data = res.json()
            st.write("🟣 Bot:", data["answer"])
        else:
            st.write("❌ Error:", res.text)
