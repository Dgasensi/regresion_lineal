import streamlit as st
from joblib import load
import numpy as np

# Carga el modelo
model = load('/workspaces/regresion_lineal/src/modelo_regresion_lineal.pkl')

def main():
    st.title('Predictor de Gastos medicos')

    # Aquí, recoges las entradas como en el ejemplo anterior
    age = st.number_input('Edad', min_value=18, max_value=64, value=30)
    bmi = st.number_input('BMI', min_value=15.0, max_value=40.0, value=28.0)
    children = st.number_input('Número de hijos', min_value=0, max_value=4, value=1)
    smoker = st.selectbox('Fumador', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Sí')
    gender = st.selectbox('Género', options=[0, 1], format_func=lambda x: 'Masculino' if x == 0 else 'Femenino')
    


    
    if st.button('Predecir'):
        X_new = np.array([[age, bmi, children, smoker, gender]])
        prediction = model.predict(X_new)
        st.write(f'Predicción: ${prediction[0]:.2f}')
        
if __name__ == '__main__':
    main()

