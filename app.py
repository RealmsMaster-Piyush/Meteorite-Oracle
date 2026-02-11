#Main brain
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize FastAPI
app = FastAPI(title="Meteorite Classifier API")

# 2. Load the model and encoder
# Make sure these files are in the same folder as this script
model = joblib.load('meteorite_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# 3. Define the input data format
class MeteoriteData(BaseModel):
    mass: float
    year: float
    lat: float
    long: float

@app.post("/predict")
def predict_rarity(data: MeteoriteData):
    # Prepare features for the model
    # Features: ['mass (g)', 'year', 'reclat', 'reclong', 'abs_lat']
    abs_lat = abs(data.lat)
    features = np.array([[data.mass, data.year, data.lat, data.long, abs_lat]])
    
    # Predict
    prediction_id = model.predict(features)[0]
    prediction_name = encoder.inverse_transform([prediction_id])[0]
    
    # Get probabilities for "Rarity Score"
    probs = model.predict_proba(features)[0]
    confidence = float(np.max(probs))

    return {
        "prediction": prediction_name,
        "confidence": f"{confidence:.2%}",
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#UI
import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Meteorite Oracle", page_icon="☄️", layout="wide")

# 2. Cinematic Comet Function
def cinematic_comet():
    """Injects a massive, detailed comet animation."""
    st.markdown(
        """
        <style>
        @keyframes comet-fly {
            0% { transform: translate(120%, -20%) scale(0.5); opacity: 0; }
            20% { opacity: 1; }
            100% { transform: translate(-150%, 100%) scale(2); opacity: 0; }
        }
        
        .comet-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 9999;
            overflow: hidden;
        }

        .giant-comet {
            position: absolute;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,1) 0%, rgba(0,212,255,1) 10%, rgba(0,50,255,0.5) 30%, transparent 70%);
            border-radius: 50%;
            filter: blur(20px);
            box-shadow: 0 0 100px 50px rgba(0,150,255,0.8);
            animation: comet-fly 2.5s ease-in forwards;
        }

        /* The tail of the comet */
        .giant-comet::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 1000px;
            height: 150px;
            background: linear-gradient(to right, rgba(255,255,255,0.8), rgba(0,150,255,0.4), transparent);
            transform: translateY(-50%) rotate(-10deg);
            transform-origin: left;
            filter: blur(10px);
        }
        </style>
        <div class="comet-container">
            <div class="giant-comet"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 3. Main UI
st.title("☄️ Meteorite Oracle: Deep Space Analysis")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔍 Input Data")
    mass = st.number_input("Mass (grams)", min_value=0.0, value=500.0, format="%.2f")
    year = st.number_input("Year Found/Fell", min_value=0, max_value=2026, value=2024)
    lat = st.slider("Latitude", -90.0, 90.0, 33.7)
    long = st.slider("Longitude", -180.0, 180.0, 130.7)
    
    predict_btn = st.button("Analyze Specimen", use_container_width=True)

with col2:
    st.header("Discovery Location")
    st.map(data={"lat": [lat], "lon": [long]}, zoom=2)

# 4. Prediction Logic
if predict_btn:
    payload = {"mass": mass, "year": year, "lat": lat, "long": long}
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        res = response.json()
        
        st.markdown("---")
        
        # LOGIC BRIDGE: Trigger comet if model says Rare OR if mass is scientifically 'Rare' (e.g., > 100kg)
        is_rare_prediction = "Iron" in res['prediction'] or "Stony-Iron" in res['prediction']
        is_massive_find = mass >= 100000.0  # 100kg+
        
        if is_rare_prediction or is_massive_find:
            cinematic_comet() 
            st.warning(f"🚨 **RARE CLASSIFICATION DETECTED**")
            # If the model was stubborn but the mass was huge, we show the truth
            display_text = res['prediction'] if is_rare_prediction else "Rare Metallic Specimen (Size Override)"
            st.subheader(f"Result: {display_text}")
        else:
            st.info(f"✅ Prediction: {res['prediction']}")
            st.write(f"Confidence: {res['confidence']}")

    except Exception as e:
        st.error("Connection Error: Is your Backend (main.py) running?")
