import streamlit as st
from diabetes import diabetes_app
from heart_disease import heart_disease_app
from parkinsons import parkinsons_app

st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")

st.title("Multiple Disease Prediction System")

st.sidebar.title("Navigation")

def sidebar_style():
    st.markdown(
        """
        <style>
            .sidebar-button {
                display: block;
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                text-align: center;
                border-radius: 5px;
                background-color: lightgray;
                color: black;
                font-size: 16px;
                cursor: pointer;
                transition: 0.3s;
                border: none;
            }
            .sidebar-button:hover {
                background-color: #4CAF50;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

sidebar_style()

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

pages = {
    "Home": lambda: st.header("Welcome to the Multiple Disease Prediction System!"),
    "Diabetes": diabetes_app,
    "Heart Disease": heart_disease_app,
    "Parkinson's Disease": parkinsons_app
}

for page_name in pages.keys():
    if st.sidebar.button(page_name, key=page_name):
        st.session_state["selected_page"] = page_name

selected_page = st.session_state["selected_page"]

if selected_page in pages:
    pages[selected_page]()
