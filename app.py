
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Carregando o modelo treinando
with open('linear_regression_model.pkl', 'rb') as f:
    lm = pickle.load(f)

# Criando o título do Aplicativo
st.title('Previsão de Preços de Casas nos EUA')

# Campos de entrada do usuário
avg_area_income = st.number_input('Média de Renda da Área', value=0.0)
avg_area_house_age = st.number_input('Média de Idade das Casas na Área', value=0.0)
avg_area_number_of_rooms = st.number_input('Média do Número de Quartos', value=0.0)
avg_area_number_of_bedrooms = st.number_input('Média do Número de Quartos de Dormir', value=0.0)
area_population = st.number_input('População da Área', value=0.0)

# Adicionando o botão de predição
if st.button('Prever Preço'):
    # Criando array numpy com valores de entrada
    user_data = np.array([[avg_area_income, avg_area_house_age, avg_area_number_of_rooms, avg_area_number_of_bedrooms, area_population]])

    # Fazendo predição com modelo carregado
    predicted_price = lm.predict(user_data)

    # Exibir o preço
    st.write(f'O preço previsto da casa é: ${predicted_price[0]:,.2f}')
