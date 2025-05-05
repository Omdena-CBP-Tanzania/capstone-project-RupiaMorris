import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = r'C:\Users\Hp\Desktop\work\capstone-project-RupiaMorris\models\random_forest.pkl'
model = joblib.load(model_path)

st.set_page_config(page_title="Climate Rainfall Prediction", layout="centered")

# App Title
st.title("🌧️ Climate Rainfall Prediction App")
st.markdown("This app predicts **Total Rainfall (mm)** based on climate conditions.")

# Sidebar Inputs
st.sidebar.header("Enter Climate Conditions")

year = st.sidebar.slider("Year", 2000, 2030, 2023)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
avg_temp = st.sidebar.number_input("Average Temperature (°C)", value=25.0)
max_temp = st.sidebar.number_input("Max Temperature (°C)", value=30.0)
min_temp = st.sidebar.number_input("Min Temperature (°C)", value=20.0)

temp_range = max_temp - min_temp

season_map = {1: "Short Rains", 2: "Dry", 3: "Long Rains", 4: "Cool Dry"}
season = st.sidebar.selectbox("Season", options=list(season_map.keys()), format_func=lambda x: season_map[x])

# Prepare input for prediction
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Average_Temperature_C': [avg_temp],
    'Max_Temperature_C': [max_temp],
    'Min_Temperature_C': [min_temp],
    'Temp_Range_C': [temp_range],
    'Season': [season]
})

# Predict and show result
if st.button("Predict Rainfall"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌧️ Predicted Total Rainfall: **{prediction:.2f} mm**")

# Optional footer
st.markdown("---")
st.caption("Capstone Project – Climate Prediction")

# ---------------------- Visualizations Section ----------------------

st.markdown("## 📊 Climate Data Visualizations")

# Load data for visualization
df = pd.read_csv(r'C:\Users\Hp\Desktop\work\capstone-project-RupiaMorris\data\preprocessed_climate_data.csv')
if 'Date' in df.columns:
    df = df.drop('Date', axis=1)

# 1. Rainfall Trend Over the Years
if st.checkbox("📈 Show Rainfall Trend Over Years"):
    yearly_rainfall = df.groupby('Year')['Total_Rainfall_mm'].mean().reset_index()
    st.line_chart(yearly_rainfall.rename(columns={'Total_Rainfall_mm': 'Average Rainfall (mm)'}).set_index('Year'))

# 2. Correlation Heatmap
if st.checkbox("🔥 Show Correlation Heatmap"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# 3. Average Rainfall by Season
if st.checkbox("🌦️ Show Average Rainfall by Season"):
    season_labels = {1: "Short Rains", 2: "Dry", 3: "Long Rains", 4: "Cool Dry"}
    df['Season_Label'] = df['Season'].map(season_labels)
    season_rainfall = df.groupby('Season_Label')['Total_Rainfall_mm'].mean().sort_values(ascending=False)

    st.bar_chart(season_rainfall)
