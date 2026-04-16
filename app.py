import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm 

st.title("Proyecto distribuciones y prueba de hipotesis")
st.write("Visualizaciones, prueba Z e integracion con la IA de Gemini")

st.header("Carga de datos")

opcion = st.radio(
    "selecciona una opcion:",
    ("Cargar CSV","Generar datos sinteticos")
)

df = None

if opcion == "Cargar CSV":
    archivo = st.file_uploader("Sube tu archivo CSV",type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.write("Vista previa de los datos:")
        st.dataframe(df)
else:
    n_sintetico = st.number_input("Numero de datos", min_value=30,value=100)
    media = st.number_input("Media",value=50)
    desviacion = st.number_input("Desviacion estandar",value=10)

    datos = np.random.normal(media,desviacion,n_sintetico)
    df = pd.DataFrame({"datos": datos})

    st.write("Datos generados")
    st.dataframe(df)

if df is not None:
    st.header("Seleccion de variable")

    columna = st.selectbox("Seleccione una columna",df.columns)
    data = df[columna].dropna()

    st.header("Visualizacion")

    #Histograma + KDE
    st.subheader("Histograma y KDE")
    fig1, ax1 = plt.subplots()
    sns.histplot(data, kde=True, ax=ax1)
    st.pyplot(fig1)

    #Boxplot
    st.subheader("Boxplot")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=data, ax=ax2)
    st.pyplot(fig2)
