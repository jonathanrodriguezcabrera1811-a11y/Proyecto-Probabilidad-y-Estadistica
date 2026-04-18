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


    st.header("Resumen estadistico")

    media = np.mean(data)
    desviacion = np.std(data)

    st.write(f"Media: {media}")
    st.write(f"Desviacion estandar: {desviacion}")

    #Analisis del usuario
    st.header("Analisis del estudiante")
    
    respuesta_usuario = st.text_area(
        "Responde:\n¿La distribución parece normal?\n¿Hay sesgo?\n¿Hay outliers?",
        height=150
    )
    st.write("Posteriormente tu respuesta sera evaluada por la IA")

    st.header("Prueba de hipotesis (Z)")

    h0 = st.number_input("Hipótesis nula (μ₀)", value=50.0)
    h1 = st.text_input("Hipótesis alternativa (ej: μ < 50)")
    sigma = st.number_input("desviacion estandar", value=10.0)
    alpha = st.number_input("Nivel de significancia", value=0.05)

    tipo = st.selectbox(
        "Tipo de prueba",
        ("Bilateral", "Cola izquierda", "Cola derecha")
    )

    if "<" in h1:
        tipo_detectado = "Cola izquierda"
    elif ">" in h1:
        tipo_detectado = "Cola derecha"
    else:
        tipo_detectado = "Bilateral"

    st.write(f"Tipo detectado por H1: {tipo_detectado}")

    if tipo != tipo_detectado:
        st.warning("ERROR: El tipo de prueba no coincide con H1")

    # H1 FORMAL
    if tipo == "Cola izquierda":
        h1_formal = f"μ < {h0}"
    elif tipo == "Cola derecha":
        h1_formal = f"μ > {h0}"
    else:
        h1_formal = f"μ ≠ {h0}"

    st.write(f"H1 formal: {h1_formal}")

    n = len(data)

    if n < 30:
        st.error("Se requiere n ≥ 30")
    else:

        #Z
        z = (media - h0) / (sigma / np.sqrt(n))

        #p-value
        if tipo == "Bilateral":
            p_value = 2 * (1 - norm.cdf(abs(z)))
            z_crit = norm.ppf(1 - alpha / 2)

        elif tipo == "Cola derecha":
            p_value = 1 - norm.cdf(z)
            z_crit = norm.ppf(1 - alpha)

        else:
            p_value = norm.cdf(z)
            z_crit = norm.ppf(alpha)

        #resultados
        st.subheader("Resultados")

        st.write(f"Z: {z:.4f}")
        st.write(f"p-value: {p_value:.6f}")

        # =========================
        # DECISION
        # =========================
        if p_value < alpha:
            decision = "RECHAZAR H0"
        else:
            decision = "NO RECHAZAR H0"

        st.success(decision)

        #Interpretación automática
        st.subheader("Interpretacion automatica")
        if p_value < alpha:
            interpretacion = (
                f"Como p-value ({p_value:.4f}) < α ({alpha}), "
                "se rechaza H0. Hay evidencia estadistica suficiente"
                "para apoyar la hipotesis alternativa"
            )
        else:
            interpretacion = (
                f"Como p-value ({p_value:.4f}) >= α ({alpha}), "
                "no se rechaza H0. No hay evidencia estadistica suficiente"
                "para apoyar la hipotesis alternativa"
            )
        st.write(interpretacion)

        #Grafica normal + Z
        st.subheader("Distribución normal y estadistico Z")

        x = np.linspace(-4,4,200)
        y = norm.pdf(x)
        
        fig3,ax3 = plt.subplots()
        ax3.plot(x,y, label="Distribucion normal")

        ax3.axvline(z, color="red", linestyle="--", label="Z observado")
        ax3.legend()

        st.pyplot(fig3)
