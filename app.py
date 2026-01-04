import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 0. DIAGNOSTICS (Visible on Web App) ---
# This block helps debug deployment issues by printing the server's file structure
st.set_page_config(page_title="UAE Rent Estimator", page_icon="🏡", layout="centered")

st.title("🏡 UAE Rent Estimator")

# Debug Expander: Check this if the app crashes!
with st.expander("⚠️ System Diagnostics (Open if Model Fails)"):
    st.write(f"**Current Working Directory:** `{os.getcwd()}`")
    st.write("**Files in this directory:**")
    try:
        files = os.listdir(os.getcwd())
        st.write(files)
        
        if 'house_price_model.pkl' in files:
            st.success("✅ Model file detected.")
        else:
            st.error("❌ Model file NOT found in root directory.")
            
    except Exception as e:
        st.error(f"Could not list files: {e}")

# -------------------------------------------

# 1. Load Model & Data
@st.cache_resource
def load_data():
    # Attempt 1: Try Absolute Path (Best for local)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'house_price_model.pkl')
    artifacts_path = os.path.join(current_dir, 'model_artifacts.pkl')
    data_path = os.path.join(current_dir, 'uae_rents_processed.csv')

    # Attempt 2: Try Relative Path (Best for Cloud if structure varies)
    if not os.path.exists(model_path):
        model_path = 'house_price_model.pkl'
        artifacts_path = 'model_artifacts.pkl'
        data_path = 'uae_rents_processed.csv'

    try:
        # Load Files
        model = joblib.load(model_path)
        artifacts = joblib.load(artifacts_path)
        df = pd.read_csv(data_path)
        return model, artifacts, df
        
    except FileNotFoundError as e:
        st.error(f"❌ File Not Found Error: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error Loading Model: {e}")
        st.warning("This is usually caused by a Scikit-Learn version mismatch. Ensure requirements.txt matches your local environment.")
        return None, None, None

model, artifacts, df = load_data()

if model is None:
    st.stop()

st.markdown("### AI-Powered Real Estate Valuation")
st.markdown("Adjust the parameters in the sidebar to estimate the annual rental price.")

# 3. Sidebar - User Inputs
# ---------------------------------------------------------
st.sidebar.header("Property Details")

# Dynamic Dropdowns
cities = sorted(df['City'].astype(str).unique())
city = st.sidebar.selectbox("City", cities)

# Filter Neighborhoods to only show those in the selected City
neighborhoods = sorted(df[df['City'] == city]['Neighborhood'].astype(str).unique())
neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods)

property_types = sorted(df['Type'].astype(str).unique())
prop_type = st.sidebar.selectbox("Property Type", property_types)

st.sidebar.markdown("---")

# Numeric Inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=7, value=2)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=7, value=2)

area = st.sidebar.slider("Area (Sq. Ft)", min_value=300, max_value=10000, value=1200, step=50)

# Binary Features (The "Premium" flags)
st.sidebar.markdown("### Features & Amenities")
is_furnished = st.sidebar.checkbox("Furnished", value=True)
has_view = st.sidebar.checkbox("Premium View (Sea/Burj)", value=False)
is_upgraded = st.sidebar.checkbox("Upgraded / Renovated", value=False)
has_pool = st.sidebar.checkbox("Private Pool", value=False)
has_maids = st.sidebar.checkbox("Maid's Room", value=False)
is_ejari = st.sidebar.checkbox("Ejari / Low Cost Unit", value=False)

# 4. Main Prediction Logic
# ---------------------------------------------------------
if st.button("Estimate Rent", type="primary"):
    
    input_data = pd.DataFrame({
        'City': [city],
        'Neighborhood': [neighborhood],
        'Type': [prop_type],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Area_SqFt': [area],
        'Is_Furnished': [1 if is_furnished else 0],
        'Is_Upgraded': [1 if is_upgraded else 0],
        'Has_View': [1 if has_view else 0],
        'Has_Maids': [1 if has_maids else 0],
        'Has_Pool': [1 if has_pool else 0],
        'Is_Ejari': [1 if is_ejari else 0]
    })

    try:
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Estimated Annual Rent")
        st.markdown(f"<h1 style='color: #4CAF50;'>AED {prediction:,.0f}</h1>", unsafe_allow_html=True)
        
        # Context: Compare to average in that neighborhood
        similar_props = df[
            (df['Neighborhood'] == neighborhood) & 
            (df['Bedrooms'] == bedrooms)
        ]
        
        if not similar_props.empty:
            avg_price = similar_props['Price_AED'].mean()
            diff = prediction - avg_price
            delta_color = "normal" if abs(diff) < 5000 else ("inverse" if diff > 0 else "off")
            
            st.metric(
                label=f"Average for {bedrooms}-Bed in {neighborhood}", 
                value=f"AED {avg_price:,.0f}", 
                delta=f"{diff:,.0f} vs Market Avg",
                delta_color=delta_color
            )
        else:
            st.info(f"Not enough data in {neighborhood} to calculate a market average comparison.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# 5. Feature Importance Chart
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📊 How does the model decide? (Feature Importance)"):
    st.write("This chart shows which factors drive the price the most based on the trained AI model.")
    
    try:
        rf_model = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']
        
        num_cols = artifacts['numeric_features']
        cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(artifacts['categorical_features'])
        
        all_feature_names = num_cols + list(cat_cols)
        importances = rf_model.feature_importances_
        
        feat_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        })
        
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
        st.bar_chart(feat_df.set_index('Feature'))
        
    except Exception as e:
        st.warning("Could not generate feature importance chart. The model structure might vary.")