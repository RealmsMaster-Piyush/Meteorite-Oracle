import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Meteorite Oracle", page_icon="☄️", layout="wide")

# 2. Load the "Brain" (Model) directly into the app
import os

# 2. Load the "Brain" (Model) with absolute paths for the cloud
@st.cache_resource
def load_model():
    # This finds the exact folder where app.py lives on the Streamlit server
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "meteorite_model.pkl")
    encoder_path = os.path.join(base_path, "label_encoder.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

# Actually call the function and assign the variables!
try:
    model, encoder = load_model()
except Exception as e:
    st.error(f"Critical Error: Could not load model files. {e}")
    st.stop() # This prevents the NameError later

# 3. Cinematic Comet Animation
def cinematic_comet():
    st.markdown(
        """
        <style>
        @keyframes comet-fly {
            0% { transform: translate(120%, -20%) scale(0.5); opacity: 0; }
            20% { opacity: 1; }
            100% { transform: translate(-150%, 100%) scale(2); opacity: 0; }
        }
        .comet-container { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 9999; overflow: hidden; }
        .giant-comet { position: absolute; width: 400px; height: 400px; background: radial-gradient(circle, #fff 0%, #00d4ff 10%, rgba(0,50,255,0.5) 30%, transparent 70%); border-radius: 50%; filter: blur(20px); box-shadow: 0 0 100px 50px rgba(0,150,255,0.8); animation: comet-fly 2.5s ease-in forwards; }
        .giant-comet::after { content: ''; position: absolute; top: 50%; left: 50%; width: 1000px; height: 150px; background: linear-gradient(to right, rgba(255,255,255,0.8), rgba(0,150,255,0.4), transparent); transform: translateY(-50%) rotate(-10deg); transform-origin: left; filter: blur(10px); }
        </style>
        <div class="comet-container"><div class="giant-comet"></div></div>
        """,
        unsafe_allow_html=True
    )

# 4. Main UI
st.title("☄️ Meteorite Oracle: Global Classifier")
st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    st.header("🔍 Input Data")
    mass = st.number_input("Mass (grams)", min_value=0.0, value=500.0)
    year = st.number_input("Year Found/Fell", min_value=0, max_value=2026, value=2024)
    lat = st.slider("Latitude", -90.0, 90.0, 33.7)
    long = st.slider("Longitude", -180.0, 180.0, 130.7)
    predict_btn = st.button("Analyze Specimen", use_container_width=True)

with col2:
    st.header("Discovery Location")
    st.map(data={"lat": [lat], "lon": [long]}, zoom=2)

# 5. Prediction logic (Now internal!)
if predict_btn:
    # Prepare features
    features = np.array([[mass, year, lat, long]])
    pred_index = model.predict(features)[0]
    prediction = encoder.inverse_transform([pred_index])[0]
    
    # Rare logic + Scientific Override for big masses
    is_rare = "Iron" in prediction or "Stony-Iron" in prediction or mass >= 100000.0
    
    st.markdown("---")
    if is_rare:
        cinematic_comet()
        st.warning(f"🚨 RARE CLASSIFICATION: {prediction.upper()}")
    else:
        st.success(f"✅ Predicted Class: {prediction}")


