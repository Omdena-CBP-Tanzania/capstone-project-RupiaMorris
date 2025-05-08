import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Set Correct Paths for Streamlit Deployment ===
model_path = 'models/random_forest.pkl'
data_path = 'data/preprocessed_climate_data.csv'

# === Load model and data safely ===
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"❌ Model file not found at: `{model_path}`. Please upload the model file.")
    st.stop()

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"❌ Data file not found at: `{data_path}`. Please upload the data file.")
    st.stop()

# === Streamlit Page Setup ===
st.set_page_config(page_title="Climate Rainfall Prediction", layout="centered")
st.title("🌧️ Climate Rainfall Prediction App")
st.markdown("This app predicts **Total Rainfall (mm)** based on the selected **Year** and **Month**.")

# === Sidebar Inputs ===
st.sidebar.header("Select Date")
year = st.sidebar.slider("Year", 2000, 2030, 2023)
month = st.sidebar.selectbox("Month", list(range(1, 13)))

# === Season Mapping ===
def get_season(month):
    if month in [3, 4, 5]:
        return 3  # Long Rains
    elif month in [10, 11, 12]:
        return 1  # Short Rains
    elif month in [6, 7, 8, 9]:
        return 4  # Cool Dry
    else:
        return 2  # Dry

season_code = get_season(month)
season_labels = {1: "Short Rains", 2: "Dry", 3: "Long Rains", 4: "Cool Dry"}
season_name = season_labels[season_code]

# === Use Averages for Other Inputs ===
avg_temp = df['Average_Temperature_C'].mean()
max_temp = df['Max_Temperature_C'].mean()
min_temp = df['Min_Temperature_C'].mean()
temp_range = max_temp - min_temp

# === Prepare Input ===
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Average_Temperature_C': [avg_temp],
    'Max_Temperature_C': [max_temp],
    'Min_Temperature_C': [min_temp],
    'Temp_Range_C': [temp_range],
    'Season': [season_code]
})

# === Make Prediction ===
if st.button("Predict Rainfall"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌧️ Predicted Total Rainfall for {year}-{month:02d} ({season_name}): **{prediction:.2f} mm**")

# === Footer ===
st.markdown("---")
st.caption("Capstone Project – Climate Prediction")

# === Visualizations ===
st.markdown("## 📊 Climate Data Visualizations")

# 1. Rainfall Trend Over Years
if st.checkbox("📈 Show Rainfall Trend Over Years"):
    yearly_rainfall = df.groupby('Year')['Total_Rainfall_mm'].mean().reset_index()
    st.line_chart(yearly_rainfall.rename(columns={'Total_Rainfall_mm': 'Average Rainfall (mm)'}).set_index('Year'))

# 2. Correlation Heatmap
if st.checkbox("🔥 Show Correlation Heatmap"):
    numeric_df = df.select_dtypes(include='number')  # Keep only numeric columns
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# 3. Average Rainfall by Season
if st.checkbox("🌦️ Show Average Rainfall by Season"):
    df['Season_Label'] = df['Season'].map(season_labels)
    season_rainfall = df.groupby('Season_Label')['Total_Rainfall_mm'].mean().sort_values(ascending=False)
    st.bar_chart(season_rainfall)