import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Konut Fiyat Tahminlemesi", layout="centered")

# Basic styling
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 4px;
        padding: 8px 16px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Konut Fiyat Tahmini")
st.write("İlçe, mahalle ve konut bilgilerinizi girerek konutuzun tavsiye edilen ilan fiyatını öğrenin.")

# Load model and data
model_path = r"C:\Users\ASUS\Desktop\Bitirme\konut_fiyat_modeli.joblib"
df_dict_path = r"C:\Users\ASUS\Desktop\Bitirme\1-data_cleaning\clean_data.csv"

try:
    model = joblib.load(model_path)
    df_dict = pd.read_csv(df_dict_path)
except:
    st.error("Model veya veri dosyası bulunamadı.")
    st.stop()

# Create encoding dictionaries
district_encoding_dict = df_dict.groupby('district')['price'].mean().to_dict()
neighbor_encoding_dict = df_dict.groupby('neighbor')['price'].mean().to_dict()

# Location inputs
st.subheader("Konum")
col1, col2 = st.columns(2)
with col1:
    ilce = st.selectbox("İlçe", sorted(district_encoding_dict.keys()))
with col2:
    mahalle = st.selectbox("Mahalle", sorted(neighbor_encoding_dict.keys()))

# Property features inputs
st.subheader("Konut Özellikleri")
col3, col4 = st.columns(2)  # Create two columns for property features

with col3:
    m2 = st.number_input("Metrekare (m2)", min_value=10, max_value=1000, value=120, step=10)
    age = st.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)
with col4:
    total_room = st.number_input("Oda Sayısı", min_value=1, max_value=40, value=3)
    floor = st.number_input("Kat Numarası", min_value=-5, max_value=50, value=2, step=1)

# Prediction
if st.button("Fiyatı Tahmin Et"):
    district_encoded = district_encoding_dict.get(ilce)
    neighbor_encoded = neighbor_encoding_dict.get(mahalle)

    if district_encoded is None or neighbor_encoded is None:
        st.error("İlçe veya mahalle encoding bulunamadı.")
    else:
        veri = pd.DataFrame([{
            "m2": m2,
            "total_room": total_room,
            "age": age,
            "floor": floor,
            "district_encoded": district_encoded,
            "neighbor_encoded": neighbor_encoded
        }])

        tahmin = model.predict(veri)[0]
        st.success(f"{ilce} - {mahalle} için tahmini fiyat: {tahmin:,.0f} TL")