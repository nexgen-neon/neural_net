import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from xor_page import show_xor_page
from grade_page import show_grade_page
from weather_page import show_weather_page


st.set_page_config(
    page_title="Neural Network Projects",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Neural Network Projects")
st.sidebar.header("Select Project")

project = st.sidebar.selectbox(
    "Choose a neural network project",
    [
        "XOR Neural Network",
        "Grade Prediction",
        "Weather Prediction",
    ],
)

if project == "XOR Neural Network":
    show_xor_page()

elif project == "Grade Prediction":
    show_grade_page()

elif project == "Weather Prediction":
    show_weather_page()