import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# 1. Page Configuration
st.set_page_config(page_title="Meteorite Oracle", page_icon="☄️", layout="wide")

# Suppress version warnings from your Python 3.14/1.8.0 mismatch
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# 2. THE BRAIN: Modular ML Loader
@st.cache_resource
def load_ml_assets():
    base_path = os.path.dirname(__file__)
    m_path = os.path.join(base_path, "meteorite_model.pkl")
    e_path = os.path.join(base_path, "label_encoder.pkl")
    
    # Check if files exist before trying to load
    if not os.path.exists(m_path) or not os.path.exists(e_path):
        return None, None, "Model files not found in the repository!"

    try:
        # Standard load
        with open(m_path, "rb") as f:
            model = pickle.load(f)
        with open(e_path, "rb") as f:
            encoder = pickle.load(f)
        return model, encoder, None
    except Exception:
        try:
            # Fallback for cross-version Python loading (Latin-1)
            with open(m_path, "rb") as f:
                model = pickle.load(f, encoding='latin1')
            with open(e_path, "rb") as f:
                encoder = pickle.load(f, encoding='latin1')
            return model, encoder, None
        except Exception as e:
            return None, None, str(e)

# 3. THE FACE: Cinematic Animation
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
        </style>
        <div class="comet-container"><div class="giant-comet"></div></div>
        """,
        unsafe_allow_html=True
    )

# 4. MAIN APP LOGIC
st.title("☄️ Meteorite Oracle: Global Classifier")
st.markdown("---")

model, encoder, error_msg = load_ml_assets()

if error_msg:
    st.error(f"❌ ML Loading Error: {error_msg}")
    st.info("Ensure your .pkl files are in the main GitHub folder.")
    st.stop()

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

if predict_btn:
    # Prepare data for the Brain
    features = np.array([[mass, year, lat, long]])
    
    # Run the Prediction
    try:
        pred_index = model.predict(features)[0]
        prediction = encoder.inverse_transform([pred_index])[0]
        
        # Check for RARE (Iron) or Massive Find
        is_rare = "Iron" in prediction or mass >= 100000.0
        
        st.markdown("---")
        if is_rare:
            cinematic_comet()
            st.warning(f"🚨 RARE CLASSIFICATION: {prediction.upper()}")
        else:
            st.success(f"✅ Predicted Class: {prediction}")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
