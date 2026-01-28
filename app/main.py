import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# --- Page Config ---
st.set_page_config(page_title="Tourism Analytics Premium", layout="wide", page_icon="✈️")

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1e0533 0%, #11032b 100%);
        color: #ffffff;
    }
    
    /* Glassmorphism Cards */
    .css-1r6slb0, .css-12oz5g7, .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #f0f2f6;
        text-align: center;
        text-shadow: 0 0 10px rgba(100, 200, 255, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255, 65, 108, 0.5);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_data.csv")

# --- Helper Functions ---
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
df = load_data()
lottie_travel = load_lottieurl("https://lottie.host/embed/9860b21e-d412-4c28-981f-79ec25256567/4w8J6X9G3I.json") # Updated URL or keep distinct
# Use a public stable URL or handle failure
# Let's use a safe check
lottie_travel = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_wys2hmsq.json")
lottie_analysis = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_t2xzjwx8.json")

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/clouds/200/000000/airplane-take-off.png", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Rating", "Predict Visit Mode", "Recommendations"])

st.sidebar.markdown("---")
st.sidebar.info("Tourism Experience Analytics v2.0\nEnhanced with AI & Premium UI")

# --- Main App ---
if df is None:
    st.error("Data processing incomplete. Please wait.")
else:
    # --- DASHBOARD ---
    if page == "Dashboard":
        col_head1, col_head2 = st.columns([3, 1])
        with col_head1:
            st.title("Travel Trends & Insights")
            st.markdown("### Explore global tourism patterns with interactive data.")
        with col_head2:
            if lottie_travel:
                st_lottie(lottie_travel, height=200, key="travel")
            else:
                st.info("✈️") # Fallback emoji if lottie fails

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Users", df['UserId'].nunique(), "+5%")
        m2.metric("Total Transactions", len(df), "+12%")
        m3.metric("Avg Rating", f"{df['Rating'].mean():.2f}", "⭐")

        st.markdown("---")
        
        # Plotly Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Ratings Distribution")
            fig = px.histogram(df, x="Rating", nbins=5, color_discrete_sequence=['#FF4B2B'], template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Top Visit Modes")
            if 'VisitMode' in df.columns:
                counts = df['VisitMode'].value_counts().reset_index()
                counts.columns = ['Mode', 'Count']
                fig2 = px.pie(counts, values='Count', names='Mode', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

        if 'Country' in df.columns:
            st.subheader("Top User Origins")
            top_countries = df['Country'].value_counts().head(10).reset_index()
            top_countries.columns = ['Country', 'Users']
            fig3 = px.bar(top_countries, x='Users', y='Country', orientation='h', color='Users', color_continuous_scale='Bluered')
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

    # --- PREDICT RATING ---
    elif page == "Predict Rating":
        st.title("AI Rating Predictor")
        st.write("Predict how a user will rate an attraction using Gradient Boosting.")
        
        try:
            reg_model = joblib.load(os.path.join(MODELS_DIR, "regression_model.pkl"))
            reg_scaler = joblib.load(os.path.join(MODELS_DIR, "regression_scaler.pkl"))
            reg_features = joblib.load(os.path.join(MODELS_DIR, "regression_features.pkl"))
            
            with st.form("rating_form"):
                col1, col2 = st.columns(2)
                with col1:
                    visit_year = st.slider("Year", 2000, 2030, 2023)
                    visit_month = st.selectbox("Month", list(range(1, 13)))
                    country_id = st.number_input("Country ID", min_value=0, value=1)
                with col2:
                    city_id = st.number_input("City ID", min_value=0, value=1)
                    type_id = st.number_input("Attraction Type ID", min_value=0, value=1)
                    mode_id = st.number_input("Visit Mode ID", min_value=0, value=1)
                
                submitted = st.form_submit_button("Predict Score")
                
                if submitted:
                    input_data = {
                        'VisitYear': visit_year, 'VisitMonth': visit_month,
                        'VisitModeId': mode_id, 'AttractionTypeId': type_id,
                        'CountryId': country_id, 'CityId': city_id, 'AttractionCityId': city_id
                    }
                    input_df = pd.DataFrame([input_data])
                    # Align cols
                    for c in reg_features:
                        if c not in input_df.columns: input_df[c] = 0
                    input_df = input_df[reg_features]
                    
                    # Scale
                    input_scaled = reg_scaler.transform(input_df)
                    pred = reg_model.predict(input_scaled)[0]
                    
                    st.success(f"Predicted Rating: {pred:.2f} / 5.0")
                    st.progress(min(pred/5, 1.0))
        except Exception as e:
            st.warning("Model not ready yet. Please ensure training is complete.")
            if lottie_analysis:
                st_lottie(lottie_analysis, height=150)

    # --- PREDICT VISIT MODE ---
    elif page == "Predict Visit Mode":
        st.title("Visit Mode Classifier")
        
        try:
            clf_model = joblib.load(os.path.join(MODELS_DIR, "classification_model.pkl"))
            clf_scaler = joblib.load(os.path.join(MODELS_DIR, "classification_scaler.pkl"))
            clf_features = joblib.load(os.path.join(MODELS_DIR, "classification_features.pkl"))
            
            with st.form("mode_form"):
                c1, c2 = st.columns(2)
                with c1:
                    year = st.number_input("Year", 2000, 2030, 2023)
                    month = st.slider("Month", 1, 12, 6)
                with c2:
                    type_id = st.number_input("Attraction Type ID", 0, 100, 1)
                    country_id = st.number_input("Country ID", 0, 200, 1)
                
                if st.form_submit_button("Classify Mode"):
                    input_data = {'VisitYear': year, 'VisitMonth': month, 'AttractionTypeId': type_id, 'CountryId': country_id, 'CityId': 0}
                    input_df = pd.DataFrame([input_data])
                    for c in clf_features:
                        if c not in input_df.columns: input_df[c] = 0
                    input_df = input_df[clf_features]
                    
                    input_scaled = clf_scaler.transform(input_df)
                    pred = clf_model.predict(input_scaled)[0]
                    st.balloons()
                    st.success(f"Predicted Visit Mode ID: **{pred}**")
                    
        except:
             st.info("Models are training... Check back soon!")

    # --- RECOMMENDATIONS ---
    elif page == "Recommendations":
        st.title("Personalized for You")
        st.markdown("Using AI to find your next adventure.")
        
        try:
            knn = joblib.load(os.path.join(MODELS_DIR, "recommendation_knn.pkl"))
            matrix = joblib.load(os.path.join(MODELS_DIR, "user_item_matrix.pkl"))
            
            user_input = st.text_input("Enter User ID", "14")
            if st.button("Discover Places"):
                try:
                    uid = int(user_input)
                    if uid in matrix.index:
                        dists, idxs = knn.kneighbors(matrix.loc[uid].values.reshape(1, -1), n_neighbors=5)
                        st.subheader("People like you visited:")
                        cols = st.columns(4)
                        for i, col in enumerate(cols):
                            # Placeholder logic for display
                            col.markdown(f"**Attraction #{idxs[0][i]}**")
                            col.image("https://source.unsplash.com/random/200x150/?travel,landmark", use_column_width=True)
                            col.caption("Highly Rated")
                    else:
                        st.warning("New user? Here are popular spots!")
                        top5 = df['AttractionId'].value_counts().head(5).index
                        st.write(f"Trending: {list(top5)}")
                except:
                    st.error("Invalid ID format")
        except:
            st.error("Recommendation engine offline.")
