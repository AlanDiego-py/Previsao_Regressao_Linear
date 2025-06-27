
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained linear regression model
with open('linear_regression_model.pkl', 'rb') as f:
    lm = pickle.load(f)

# Create a title for the Streamlit app
st.title('Previsão de Preços de Casas nos EUA')

# Create input fields for user data
avg_area_income = st.number_input('Média de Renda da Área', value=0.0)
avg_area_house_age = st.number_input('Média de Idade das Casas na Área', value=0.0)
avg_area_number_of_rooms = st.number_input('Média do Número de Quartos', value=0.0)
avg_area_number_of_bedrooms = st.number_input('Média do Número de Quartos de Dormir', value=0.0)
area_population = st.number_input('População da Área', value=0.0)

# Add a button to trigger prediction
if st.button('Prever Preço'):
    # Create a NumPy array with user input values
    user_data = np.array([[avg_area_income, avg_area_house_age, avg_area_number_of_rooms, avg_area_number_of_bedrooms, area_population]])

    # Make a prediction using the loaded model
    predicted_price = lm.predict(user_data)

    # Display the predicted price
    st.write(f'O preço previsto da casa é: ${predicted_price[0]:,.2f}')
