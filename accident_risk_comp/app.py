# robust app.py — use with final_gradient_boosting_model.pkl and (ideally) model_features.pkl
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(page_title="Accident Risk Predictor (Robust)", layout="centered")
st.title("🚦 Accident Risk Prediction (Robust)")

MODEL_PATH = "final_gradient_boosting_model.pkl"
FEATURES_PATH = "model_features.pkl"

# ---------- Helpers ----------
def try_load_model(path):
    """Load model and return (model, errmsg). errmsg None on success."""
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, str(e)

def try_load_features(path):
    try:
        feats = joblib.load(path)
        return feats, None
    except Exception as e:
        return None, str(e)

def infer_features_from_model(model):
    """
    Try several ways to obtain feature names from the fitted model:
    - model.feature_names_in_ (sklearn >=1.0 if fitted on DataFrame)
    - for XGBoost/Booster: booster.feature_names
    - as last resort: None
    """
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    # XGBoost/sklearn wrapper
    try:
        booster = getattr(model, "get_booster", None)
        if booster is not None:
            b = model.get_booster()
            if hasattr(b, "feature_names") and b.feature_names is not None:
                return list(b.feature_names)
    except Exception:
        pass
    return None

def safe_align_input(user_df, model_features):
    """
    Create one-hot dummies for known categorical columns from user input
    and reindex to model_features. Fill missing with 0 and ensure numeric.
    """
    # One-hot encode everything user supplied (safe)
    df_enc = pd.get_dummies(user_df)
    # Add any missing cols
    for c in model_features:
        if c not in df_enc.columns:
            df_enc[c] = 0
    # Keep only model_features (order matters)
    df_final = df_enc[model_features].astype(float)
    # Fill any NaN with median/0
    df_final = df_final.fillna(df_final.median(numeric_only=True)).fillna(0)
    return df_final

# ---------- Load model & features (robustly) ----------
model, model_features = None, None
model_err = None
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Put your trained model in this folder.")
    st.stop()

model, model_err = try_load_model(MODEL_PATH)
if model_err:
    st.error("Failed to load model. See details below and follow the instructions.")
    st.code(model_err)
    st.markdown("""
    *Common fixes*
    - Ensure you're using the same scikit-learn version used for training.  
      Example: pip install scikit-learn==1.2.2
    - If you trained with a notebook, re-save with joblib.dump(best_model, 'final_gradient_boosting_model.pkl')
    """)
    st.stop()

# Try loading model_features.pkl, else infer
if os.path.exists(FEATURES_PATH):
    model_features, feats_err = try_load_features(FEATURES_PATH)
    if feats_err:
        st.warning("model_features.pkl exists but couldn't be loaded. Will try to infer features from model.")
        model_features = None
else:
    model_features = None

if model_features is None:
    inferred = infer_features_from_model(model)
    if inferred is not None:
        model_features = inferred
        st.info("Feature names inferred from the model object (automatic).")
    else:
        st.warning("Could not infer feature names from the model. You must create and save 'model_features.pkl' from your training notebook.")
        st.markdown("### How to create model_features.pkl (run in your training notebook)")
        st.code("""
# After you have X_train (the DataFrame used to fit the model) and your trained model:
import joblib
feature_names = list(X_train.columns)   # IMPORTANT: X_train must be the DataFrame used for fitting
joblib.dump(feature_names, "model_features.pkl")
# Then, re-save the model (in same environment) with:
joblib.dump(best_gb, "final_gradient_boosting_model.pkl")
""")
        st.stop()

# At this point model and model_features exist
st.success("Model and feature-list loaded successfully.")

# ---------- UI inputs ----------
st.subheader("Enter road & environment details (then click Predict)")

col1, col2 = st.columns(2)
with col1:
    road_type = st.selectbox("Road Type", ["urban","rural","highway"])
    lighting = st.selectbox("Lighting", ["daylight","dim","night"])
    weather = st.selectbox("Weather", ["clear","rainy","foggy","snowy"])
    time_of_day = st.selectbox("Time of Day", ["morning","afternoon","evening","night"])
with col2:
    num_lanes = st.slider("Number of Lanes", 1, 8, 2)
    curvature = st.number_input("Curvature (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    speed_limit = st.number_input("Speed Limit (km/h)", min_value=10, max_value=200, value=60)
    num_reported_accidents = st.number_input("Reported Accidents (past year)", min_value=0, max_value=100, value=1)

road_signs_present = st.checkbox("Road signs present", value=True)
public_road = st.checkbox("Public road", value=True)
holiday = st.checkbox("Holiday", value=False)
school_season = st.checkbox("School season active", value=True)

# Build raw input DataFrame
raw = pd.DataFrame([{
    "road_type": road_type,
    "lighting": lighting,
    "weather": weather,
    "time_of_day": time_of_day,
    "num_lanes": num_lanes,
    "curvature": curvature,
    "speed_limit": speed_limit,
    "num_reported_accidents": num_reported_accidents,
    "road_signs_present": int(road_signs_present),
    "public_road": int(public_road),
    "holiday": int(holiday),
    "school_season": int(school_season)
}])

st.write("Input preview:")
st.dataframe(raw.T)

# ---------- Predict button ----------
if st.button("🔍 Predict Accident Risk"):
    try:
        # Z-cap numeric outliers (safe)
        num_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents"]
        for c in num_cols:
            if c in raw.columns:
                mu = raw[c].mean()
                sigma = raw[c].std() if raw[c].std() != 0 else 1.0
                raw[c] = np.clip(raw[c], mu - 3*sigma, mu + 3*sigma)

        # Align user input into model features
        X_input = pd.get_dummies(raw)
        # Ensure all model_features present
        X_input = X_input.reindex(columns=model_features, fill_value=0)
        # Convert to numeric and fill any NaN
        X_input = X_input.astype(float).fillna(0)

        # Final check: no NaNs
        if X_input.isnull().any().any():
            st.error("Preprocessing produced NaNs. Cannot predict. Re-check feature list.")
            st.write(X_input.isnull().sum())
        else:
            ypred = model.predict(X_input)[0]
            # clip if you expect probability-like output 0-1
            try:
                ypred = float(ypred)
            except:
                pass
            st.subheader("Prediction")
            st.metric("Accident risk (score)", f"{ypred:.4f}")

            # Simple interpretation
            if isinstance(ypred, (int,float)):
                if ypred < 0 or ypred > 1:
                    # it's not probability, but still show ranges relative to training range if known
                    st.info("Note: model output not constrained to [0,1]; interpret per your training target scale.")
                if ypred < 0.2:
                    st.success("Low risk")
                elif ypred < 0.5:
                    st.warning("Moderate risk")
                else:
                    st.error("High risk")
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.code(traceback.format_exc())
        # Helpful instructions:
        st.markdown("""
        *If error mentions missing feature names*:
        - Ensure model_features.pkl exists and was created from the exact DataFrame columns used for training:
          
          joblib.dump(list(X_train.columns), "model_features.pkl")
          
        *If error mentions sklearn internal module issues (pickle/unpickle)*:
        - Re-save the model in the same environment you will run the app:
          
          joblib.dump(best_gb, "final_gradient_boosting_model.pkl")
          
        - Or install the older sklearn (example): pip install scikit-learn==1.2.2
        """)