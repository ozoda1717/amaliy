import streamlit as st
import joblib
import pandas as pd

st.title("Tirbandlikni bashorat qilish")

# Foydalanuvchidan ma'lumotlarni kiritishni so'rash
Date = st.number_input("Sanani kiriting",min_value=1,max_value=31, step=1)

options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
Day_of_the_week = st.selectbox("Hafta kunini tanlang:", options)
CarCount = st.number_input("Moshina soni", min_value=18,max_value=163)
BikeCount= st.number_input("Velosipedlar soni", min_value=5,max_value=57, step=1)
BusCount= st.number_input("Avtobus soni", min_value=0, max_value=47, step=1)
TruckCount= st.number_input("Yuk moshina soni", min_value=5,max_value=40, step=1)
Total = st.number_input("Umumiy soni", min_value=21,max_value=278, step=1)


# Modelni yuklash va bashorat qilish
if st.button("Tirbandlikni bashorat  qilish"):
    # Kiritilgan ma'lumotlarni DataFrame ga o'tkazish
    input_data = {
        "Date": [Date],
        "Day_of_the_week": [Day_of_the_week],
        "CarCount": [CarCount],
        "BikeCount": [BikeCount],
        "BusCount": [BusCount],
        "TruckCount": [TruckCount],
        "Total": [Total],
        

    }
    
    df = pd.DataFrame(input_data)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['Day_of_the_week'])

    # Modelni yuklash
    model = joblib.load('decision_tree_model (12).pkl')  # Model faylingiz nomini mos ravishda kiriting

    # Bashorat qilish
    outcome = model.predict(df_encoded)

    # Natijani ko'rsatish
    st.write(f"Bashorat qilingan sifat: {outcome[0]}")