import os
import pickle
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Config ----------
FEATURE_ORDER = [
    'pl_pnum', 'ra', 'dec', 'pl_tranmid', 'pl_trandurh', 'pl_trandep',
    'st_tmag', 'st_tmagerr1', 'st_tmagerr2'
]

FEATURE_DESCRIPTIONS = {
    'pl_pnum': 'The planet number of the target star (e.g., the first planet in the system).',
    'ra': 'Right Ascension (RA) in degrees, the angular distance along the celestial equator.',
    'dec': 'Declination (DEC) in degrees, the angular distance north or south of the celestial equator.',
    'pl_tranmid': 'Mid-transit time (days) when the planet transits in front of its star.',
    'pl_trandurh': 'Duration of the planet’s transit (hours).',
    'pl_trandep': 'The depth of the planet’s transit (in parts per thousand, ppt).',
    'st_tmag': 'The apparent magnitude of the star (brightness observed from Earth).',
    'st_tmagerr1': 'The uncertainty in the star’s apparent magnitude measurement (first error margin).',
    'st_tmagerr2': 'The uncertainty in the star’s apparent magnitude measurement (second error margin).'
}

LABEL_MAP = {0: 'False Positive (FP)', 1: 'Planet Candidate (PC)'}

MODEL_PATH = "best_model_Gradient_Boosting.pkl"
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="TOI Classifier (FP vs PC)", layout="centered")

st.title("TOI Classifier: False Positive (FP) vs Planet Candidate (PC)")

st.caption("""
This model predicts whether a **TESS Object of Interest (TOI)** is a **False Positive (FP)** or a **Planet Candidate (PC)**.
The model uses a **Gradient Boosting Classifier** trained on **StandardScaler**-transformed features.
Please enter the values below to make a prediction.
""")

# ---------- Load artifacts ----------
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)  # pipeline with only 'clf' step
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

def ensure_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with required features in order, cast to float."""
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURE_ORDER].astype(float)

def predict_df(model, scaler, X: pd.DataFrame) -> pd.DataFrame:
    """Scale X, run prediction, and return a result DataFrame."""
    X_scaled = scaler.transform(X)
    y_prob = getattr(model, "predict_proba", lambda Z: None)(X_scaled)
    y_pred = model.predict(X_scaled)

    out = pd.DataFrame({
        "prediction_int": y_pred,
        "prediction": [LABEL_MAP[int(v)] for v in y_pred]
    })

    if y_prob is not None:
        # Check classes to align probabilities
        classes_ = getattr(model, "classes_", None)
        if classes_ is None and hasattr(model, "named_steps"):
            last_step = model.named_steps.get("clf", model)
            classes_ = getattr(last_step, "classes_", None)
        if classes_ is not None:
            classes_ = list(classes_)
            try:
                idx_pc = classes_.index(1)
                idx_fp = classes_.index(0)
                out["proba_PC"] = y_prob[:, idx_pc]
                out["proba_FP"] = y_prob[:, idx_fp]
            except Exception:
                if y_prob.shape[1] == 2:
                    out["proba_PC"] = y_prob[:, 1]
                    out["proba_FP"] = y_prob[:, 0]
        else:
            if y_prob is not None and y_prob.shape[1] == 2:
                out["proba_PC"] = y_prob[:, 1]
                out["proba_FP"] = y_prob[:, 0]
    return out

# Load model and scaler
try:
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    st.success("Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

# Tabs for single and batch input
tab_single, tab_batch = st.tabs(["🔢 Single Input", "📄 Batch (CSV)"])

# Single input prediction
with tab_single:
    st.subheader("Enter Feature Values")
    
    # Create a form to group the inputs and submit button
    with st.form(key="features_form"):
        # Creating 3 columns per row for feature input
        cols = st.columns(3)

        values = {}

        # Loop through the features and add each one in the appropriate column
        for i, feat in enumerate(FEATURE_ORDER):
            with cols[i % 3]:  # Ensures 3 features per row
                # Input for each feature
                values[feat] = st.number_input(feat,
                                               value=0.0,
                                               format="%.9f")
                # Feature description below the input field
                st.markdown(f"*{FEATURE_DESCRIPTIONS[feat]}*")

        # Submit button to make predictions
        submit_button = st.form_submit_button(label="Predict")

    # Prediction on single input
    if submit_button:
        try:
            # Create a DataFrame with correct columns in the correct order
            X = pd.DataFrame([values])[FEATURE_ORDER]  # Ensure the columns are in the same order as FEATURE_ORDER
            X = ensure_feature_frame(X)  # Ensures that the dataframe matches the model's expected input

            # Make predictions
            res = predict_df(model, scaler, X)

            # Extract the class prediction
            lab = res.loc[0, "prediction"]

            # Display the result
            st.markdown(f"### Result: **{lab}**")

            # Check if probabilities are available
            if "proba_PC" in res.columns and "proba_FP" in res.columns:
                st.write("Probabilities:")
                st.write(f"Planet Candidate (PC) Probability: {res['proba_PC'].values[0]:.4f}")
                st.write(f"False Positive (FP) Probability: {res['proba_FP'].values[0]:.4f}")
            else:
                st.write("No probabilities available for this prediction.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Batch CSV prediction
# Batch CSV prediction
with tab_batch:
    st.subheader("Upload a CSV")
    st.caption("CSV must contain the following columns (exact names): " + ", ".join(FEATURE_ORDER))
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if up is not None:
        try:
            # Read the uploaded CSV
            df_uploaded = pd.read_csv(up)
            st.write("Preview:", df_uploaded.head())
            
            # Ensure the required columns are present in the uploaded file
            if all(col in df_uploaded.columns for col in FEATURE_ORDER):
                # Process the features
                X = ensure_feature_frame(df_uploaded)
                res = predict_df(model, scaler, X)
                
                st.write("Predictions:")
                result = pd.concat([df_uploaded.reset_index(drop=True), res], axis=1)
                st.dataframe(result)

                # Allow the user to download the prediction results
                st.download_button(
                    "Download predictions CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error("Uploaded CSV is missing required columns.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
