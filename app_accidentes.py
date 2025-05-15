
# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import streamlit as st

# Cargar el modelo actualizado
filename = "C:/Users/Luz Marcela/Downloads/modelo-clas-hiper.pkl"

modelTree, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# Configuración básica de la app
st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=100)
st.title('Predicción de Gravedad de Accidentes de Tránsito - Envigado')
st.markdown('Ingrese los datos relevantes del accidente para predecir su gravedad.')

# Sección de entrada de datos
st.sidebar.title("Datos del accidente")

estado_beodez = st.sidebar.selectbox('¿Estado de embriaguez?', [0, 1])
resultado_beodez = st.sidebar.selectbox('Resultado de prueba de embriaguez', [0, 1])
dia_semana = st.sidebar.selectbox('Día de la semana', ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
clase_accidente = st.sidebar.selectbox('Clase de accidente', ['Choque', 'Atropello', 'Volcamiento', 'Otro'])
causa = st.sidebar.selectbox('Causa probable', ['No respetó señales', 'Exceso de velocidad', 'Alcohol', 'Otro'])

# Construcción del DataFrame de entrada
entrada = {
    'ESTADO DE BEODEZ': [estado_beodez],
    'RESULTADO DE BEODEZ': [resultado_beodez],
    f'DÍA DE LA SEMANA_{dia_semana}': [1],
    f'CLASE DE ACCIDENTE_{clase_accidente}': [1],
    f'CAUSA_{causa}': [1]
}

df_entrada = pd.DataFrame(entrada)
df_entrada = df_entrada.reindex(columns=variables, fill_value=0)

# Normalización de las variables numéricas
df_entrada[['ESTADO DE BEODEZ', 'RESULTADO DE BEODEZ']] = min_max_scaler.transform(
    df_entrada[['ESTADO DE BEODEZ', 'RESULTADO DE BEODEZ']]
)

# Botón de predicción
if st.sidebar.button("Predecir"):
    prediccion = modelTree.predict(df_entrada)
    resultado = labelencoder.inverse_transform(prediccion)
    st.subheader("Resultado de la predicción")
    st.success(f"**Gravedad predicha:** {resultado[0]}")
else:
    st.caption("*Selecciona las variables y haz clic en Predecir*")
