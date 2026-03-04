import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Smart Burnout System", layout="centered")


st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #2c3e50, #4a657a);
        color: #ffffff;
    }
    .big-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ecf0f1;
    }
    .center-text {
        text-align: center;
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 18px;
        border-radius: 10px;
        background-color: #1abc9c;
        color: #ffffff;
    }
    .stButton>button:hover {
        background-color: #16a085;
    }
    </style>
""", unsafe_allow_html=True)


data = pd.read_csv("dataset.csv")

le_mood = LabelEncoder()
le_burnout = LabelEncoder()

data["mood"] = le_mood.fit_transform(data["mood"])
data["burnout"] = le_burnout.fit_transform(data["burnout"])

X = data[["study_hours", "sleep_hours", "break_frequency", "stress_level", "mood"]]
y = data["burnout"]

model = DecisionTreeClassifier()
model.fit(X, y)


if "page" not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":

    st.markdown('<div class="big-title">AI-Based Smart Study Break & Burnout Detection System</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="center-text">A smart system to monitor study habits and suggest healthy study breaks.</div>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    if st.button("Proceed to Assessment"):
        st.session_state.page = "form"


elif st.session_state.page == "form":

    st.header("Enter Your Study Details")

    study_hours = st.number_input("Study Hours per Day", 0.0, 15.0)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0)
    break_frequency = st.number_input("Break Frequency per Day", 0.0, 10.0)
    stress_level = st.slider("Stress Level (1-10)", 1, 10)
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Stressed"])

    if st.button("Predict Burnout"):
        mood_encoded = le_mood.transform([mood])[0]
        input_data = [[study_hours, sleep_hours, break_frequency, stress_level, mood_encoded]]
        prediction = model.predict(input_data)
        burnout_level = le_burnout.inverse_transform(prediction)[0]

        st.session_state.burnout = burnout_level
        st.session_state.page = "result"


elif st.session_state.page == "result":

    burnout_level = st.session_state.burnout

    st.header("Assessment Result")

    st.subheader(f"Your Burnout Level: {burnout_level}")

    if burnout_level == "High":
        st.error("⚠️ High Burnout Detected")
        st.write("""
- Take a complete 40–60 minute break away from screens.  
- Hydrate well (at least 2 glasses of water).  
- Do light stretching.  
- Avoid studying for the next hour.  
- Ensure 7–8 hours of sleep.
""")

    elif burnout_level == "Medium":
        st.warning("⚡ Moderate Burnout Level")
        st.write("""
- Take a 15–20 minute break.  
- Drink water (1–2 glasses).  
- Stretch or walk for 5 minutes.  
- Resume study with short 45-minute sessions.
""")

    else:
        st.success("✅ Low Burnout Level")
        st.write("""
- Your routine is balanced.  
- Continue with small breaks every hour.  
- Stay hydrated and get good sleep.
""")

    st.write("")
    if st.button("Start New Assessment"):
        st.session_state.page = "home"