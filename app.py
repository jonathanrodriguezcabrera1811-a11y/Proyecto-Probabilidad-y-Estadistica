import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis
import google.generativeai as genai


st.set_page_config(
    page_title="Analisis Estadistico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0b0d12;
    --surf:    #111520;
    --border:  #1c2438;
    --accent:  #38bdf8;
    --text:    #dce6f5;
    --muted:   #4e5f7a;
    --success: #34d399;
    --danger:  #f87171;
    --gold:    #fbbf24;
}

html, body, [data-testid="stApp"] {
    background: var(--bg);
    font-family: 'Outfit', sans-serif;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.site-header {
    padding: 52px 0 40px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 44px;
}
.site-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.9rem;
    color: var(--text);
    letter-spacing: -0.5px;
    margin: 0;
}
.site-title em { color: var(--accent); font-style: normal; }
.site-sub {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 10px;
}

[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border);
    gap: 0;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 14px 30px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.18s !important;
}
button[data-baseweb="tab"]:hover { color: var(--text) !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabPanel"] { padding-top: 40px; }

.slabel {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
}
.stitle {
    font-family: 'DM Serif Display', serif;
    font-size: 1.65rem;
    color: var(--text);
    margin: 0 0 24px;
}

.note {
    border-left: 2px solid var(--accent);
    background: rgba(56,189,248,0.04);
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #8aafc8;
    margin: 10px 0 22px;
    line-height: 1.65;
}

.sblock {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
}
.sval {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--accent);
    line-height: 1;
}
.slbl {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 7px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.card {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 26px 28px;
    margin-bottom: 18px;
}
.card-sm {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
}

.verdict-r {
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.32);
    border-radius: 10px;
    padding: 22px 28px;
    text-align: center;
}
.verdict-r .vl {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--danger);
}
.verdict-r .vs {
    font-size: 0.8rem;
    color: rgba(248,113,113,0.65);
    margin-top: 6px;
}
.verdict-ok {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.32);
    border-radius: 10px;
    padding: 22px 28px;
    text-align: center;
}
.verdict-ok .vl {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--success);
}
.verdict-ok .vs {
    font-size: 0.8rem;
    color: rgba(52,211,153,0.65);
    margin-top: 6px;
}

.hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

.stSelectbox label, .stNumberInput label, .stTextArea label, .stRadio label {
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: var(--muted) !important;
}
input, textarea, select {
    background: var(--surf) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

div.stButton > button {
    background: var(--accent);
    color: #050c14;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.86rem;
    letter-spacing: 0.04em;
    transition: opacity 0.18s;
}
div.stButton > button:hover { opacity: 0.8; background: var(--accent); }

[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: var(--accent) !important;
}

.pgfooter {
    text-align: center;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    padding: 40px 0 16px;
    border-top: 1px solid var(--border);
    margin-top: 64px;
}
</style>
""", unsafe_allow_html=True)

# Configuración Gemini
GEMINI_API_KEY = "AIzaSyDQ5xwJ1fDcv_eusW5np3ABpCAOvVoZAPA"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

C_BG   = "#0b0d12"
C_SURF = "#111520"
C_GRID = "#1c2438"
C_TEXT = "#dce6f5"
C_MUTE = "#4e5f7a"
C_BLUE = "#38bdf8"
C_GOLD = "#fbbf24"
C_GRN  = "#34d399"
C_RED  = "#f87171"


def new_fig(w=9, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_SURF)
    ax.tick_params(colors=C_MUTE, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    ax.xaxis.label.set_color(C_MUTE)
    ax.yaxis.label.set_color(C_MUTE)
    ax.title.set_color(C_TEXT)
    ax.grid(color=C_GRID, linewidth=0.55, linestyle="--", alpha=0.7)
    return fig, ax


st.markdown("""
<div class="site-header">
    <h1 class="site-title">Analisis <em>Estadistico</em></h1>
    <p class="site-sub">Distribuciones &nbsp;·&nbsp; Prueba de Hipotesis &nbsp;·&nbsp; Asistente IA</p>
</div>
""", unsafe_allow_html=True)


for k in ("df", "data", "col", "media", "mediana", "desv",
          "val_sesgo", "val_kurt", "n", "q1", "q3", "n_out",
          "z_stat", "p_value", "z_crit", "alpha", "sigma",
          "h0", "tipo", "decision", "reg_crit", "h1_str",
          "resp_normal", "resp_sesgo", "resp_outliers", "resp_texto"):
    if k not in st.session_state:
        st.session_state[k] = None

tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Distribucion", "Prueba Z", "Asistente IA"])

# ─────────────────────────────────────────────
#  TAB 1 — DATOS
# ─────────────────────────────────────────────
with tab1:
    st.markdown('<p class="slabel">Paso 1</p><h2 class="stitle">Carga de datos</h2>', unsafe_allow_html=True)

    mode = st.radio(
        "Fuente",
        ("Archivo CSV", "Datos sinteticos"),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if mode == "Archivo CSV":
        st.markdown('<p class="note">Sube un archivo .csv con al menos una columna numerica. La primera fila debe contener los nombres de las variables.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Seleccionar archivo", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df_new = pd.read_csv(uploaded)
            st.session_state["df"] = df_new
            st.success(f"Archivo cargado — {df_new.shape[0]} filas, {df_new.shape[1]} columnas")
            st.dataframe(df_new, use_container_width=True, height=350)
    else:
        st.markdown('<p class="note">Genera una muestra aleatoria con distribucion normal. Define los parametros y presiona el boton para crear los datos.</p>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            n_sin = st.number_input("Tamano de muestra (n)", min_value=30, value=120)
        with g2:
            mu_sin = st.number_input("Media (mu)", value=50.0)
        with g3:
            sd_sin = st.number_input("Desviacion estandar (sigma)", value=10.0, min_value=0.01)

        if st.button("Generar muestra"):
            arr = np.random.normal(mu_sin, sd_sin, int(n_sin))
            df_new = pd.DataFrame({"valor": arr})
            st.session_state["df"] = df_new
            st.success(f"Muestra generada — {int(n_sin)} observaciones, Normal(mu={mu_sin}, sigma={sd_sin})")
            st.dataframe(df_new, use_container_width=True, height=350)

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.error("El archivo no contiene columnas numericas.")
        else:
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown('<p class="slabel">Variable de analisis</p>', unsafe_allow_html=True)
            col_sel = st.selectbox("Selecciona la columna", num_cols, label_visibility="collapsed")

            data = df[col_sel].dropna()
            if len(data) < 30:
                st.error("Se necesitan al menos 30 observaciones para la prueba Z.")
            else:
                media     = float(np.mean(data))
                mediana   = float(np.median(data))
                desv      = float(np.std(data, ddof=1))
                val_sesgo = float(skew(data))
                val_kurt  = float(kurtosis(data))
                n         = len(data)
                q1        = float(np.percentile(data, 25))
                q3        = float(np.percentile(data, 75))
                iqr_v     = q3 - q1
                n_out     = int(((data < q1 - 1.5*iqr_v) | (data > q3 + 1.5*iqr_v)).sum())

                for k, v in [("data", data), ("col", col_sel), ("media", media),
                              ("mediana", mediana), ("desv", desv), ("val_sesgo", val_sesgo),
                              ("val_kurt", val_kurt), ("n", n), ("q1", q1),
                              ("q3", q3), ("n_out", n_out)]:
                    st.session_state[k] = v

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown('<p class="slabel">Resumen estadistico</p>', unsafe_allow_html=True)

                sc = st.columns(6)
                lbls = ["Media", "Mediana", "Desv. Est.", "Sesgo", "Curtosis", "n"]
                vals = [f"{media:.3f}", f"{mediana:.3f}", f"{desv:.3f}",
                        f"{val_sesgo:.3f}", f"{val_kurt:.3f}", str(n)]
                for col, lb, vl in zip(sc, lbls, vals):
                    with col:
                        st.markdown(f"""
<div class="sblock">
    <div class="sval">{vl}</div>
    <div class="slbl">{lb}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.info("Datos listos. Navega a las siguientes pestanas para continuar.")


# ─────────────────────────────────────────────
#  TAB 2 — DISTRIBUCION
# ─────────────────────────────────────────────
with tab2:
    st.markdown('<p class="slabel">Paso 2</p><h2 class="stitle">Visualizacion de la distribucion</h2>', unsafe_allow_html=True)

    if st.session_state["data"] is None:
        st.warning("Primero carga tus datos en la pestana Datos.")
    else:
        data      = st.session_state["data"]
        media     = st.session_state["media"]
        mediana   = st.session_state["mediana"]
        val_sesgo = st.session_state["val_sesgo"]
        val_kurt  = st.session_state["val_kurt"]
        n_out     = st.session_state["n_out"]
        q1        = st.session_state["q1"]
        q3        = st.session_state["q3"]

        cg1, cg2 = st.columns(2)

        with cg1:
            st.markdown('<p class="slabel">Histograma y KDE</p>', unsafe_allow_html=True)
            st.markdown('<p class="note">Las barras muestran la frecuencia de cada rango de valores. La curva suavizada estima la forma de la distribucion. Una campana simetrica es caracteristica de la distribucion normal.</p>', unsafe_allow_html=True)

            fig1, ax1 = new_fig(7, 4.2)
            ax1.hist(data, bins="auto", density=True,
                     color=C_BLUE, alpha=0.22, edgecolor=C_SURF, linewidth=0.3)

            kde_ax = ax1.twinx()
            kde_ax.set_facecolor("none")
            kde_ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
            for sp in kde_ax.spines.values():
                sp.set_visible(False)
            sns.kdeplot(data, ax=kde_ax, color=C_BLUE, linewidth=2.2)

            ax1.axvline(media,   color=C_GOLD, linewidth=1.7, linestyle="--",
                        label=f"Media   {media:.2f}")
            ax1.axvline(mediana, color=C_GRN,  linewidth=1.7, linestyle=":",
                        label=f"Mediana {mediana:.2f}")
            ax1.set_xlabel("Valores")
            ax1.set_ylabel("Densidad")
            ax1.set_title("Histograma + KDE", fontsize=10.5, fontweight="bold", pad=12)
            ax1.legend(facecolor=C_SURF, labelcolor=C_TEXT, fontsize=8.5,
                       framealpha=0.9, edgecolor=C_GRID)
            st.pyplot(fig1, use_container_width=True)

        with cg2:
            st.markdown('<p class="slabel">Boxplot</p>', unsafe_allow_html=True)
            st.markdown('<p class="note">La caja abarca el 50% central (Q1 a Q3). Los puntos fuera de los bigotes son valores atipicos. Una caja descentrada respecto a la mediana indica asimetria.</p>', unsafe_allow_html=True)

            fig2, ax2 = new_fig(7, 4.2)
            ax2.boxplot(
                data, vert=False, patch_artist=True,
                medianprops=dict(color=C_GOLD, linewidth=2.2),
                whiskerprops=dict(color=C_MUTE, linewidth=1.3, linestyle="--"),
                capprops=dict(color=C_MUTE, linewidth=1.8),
                flierprops=dict(marker="o", color=C_RED, markersize=5,
                                markerfacecolor=C_RED, alpha=0.75, markeredgewidth=0),
                boxprops=dict(facecolor="#38bdf810", linewidth=1.3, edgecolor=C_BLUE),
            )
            for val, lbl, clr in [(q1, f"Q1 = {q1:.2f}", C_MUTE),
                                   (mediana, f"Med = {mediana:.2f}", C_GOLD),
                                   (q3, f"Q3 = {q3:.2f}", C_MUTE)]:
                ax2.text(val, 1.44, lbl, ha="center", va="bottom",
                         color=clr, fontsize=7.5, fontfamily="monospace")
            ax2.set_xlabel("Valores")
            ax2.set_yticks([])
            ax2.set_title("Boxplot", fontsize=10.5, fontweight="bold", pad=12)
            st.pyplot(fig2, use_container_width=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Observaciones de los datos</p>', unsafe_allow_html=True)
        ob1, ob2, ob3 = st.columns(3)
        with ob1:
            if abs(val_sesgo) < 0.5:
                st.success(f"Simetria aceptable — sesgo = {val_sesgo:.3f}")
            elif val_sesgo > 0.5:
                st.warning(f"Sesgo positivo {val_sesgo:.3f} — cola derecha")
            else:
                st.warning(f"Sesgo negativo {val_sesgo:.3f} — cola izquierda")
        with ob2:
            if abs(val_kurt) < 1:
                st.success(f"Curtosis normal — {val_kurt:.3f}")
            else:
                st.warning(f"Curtosis inusual — {val_kurt:.3f}")
        with ob3:
            if n_out == 0:
                st.success("Sin outliers detectados")
            else:
                st.warning(f"{n_out} outlier(s) detectados")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Tu analisis</p>', unsafe_allow_html=True)
        st.markdown('<p class="note">Responde con base en lo que observas en las graficas. Tu analisis sera evaluado por la IA en la ultima pestana.</p>', unsafe_allow_html=True)

        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            rn = st.selectbox(
                "La distribucion parece normal?",
                ["Sin respuesta", "Si, parece normal",
                 "No parece normal", "No estoy seguro"],
                key="rn_sel",
            )
        with sa2:
            rs = st.selectbox(
                "Hay sesgo en los datos?",
                ["Sin respuesta", "No hay sesgo", "Sesgo positivo",
                 "Sesgo negativo", "No puedo determinarlo"],
                key="rs_sel",
            )
        with sa3:
            ro = st.selectbox(
                "Hay outliers?",
                ["Sin respuesta", "Si, hay outliers",
                 "No hay outliers", "Posibles valores atipicos"],
                key="ro_sel",
            )

        texto = st.text_area(
            "Razonamiento breve",
            placeholder="Describe lo que observas en el histograma y el boxplot...",
            height=95,
            key="texto_sel",
        )

        if st.button("Guardar analisis"):
            st.session_state["resp_normal"]   = rn
            st.session_state["resp_sesgo"]    = rs
            st.session_state["resp_outliers"] = ro
            st.session_state["resp_texto"]    = texto
            st.success("Analisis guardado. Puedes continuar con la prueba Z.")


# ─────────────────────────────────────────────
#  TAB 3 — PRUEBA Z
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<p class="slabel">Paso 3</p><h2 class="stitle">Prueba de hipotesis Z</h2>', unsafe_allow_html=True)

    if st.session_state["data"] is None:
        st.warning("Primero carga tus datos en la pestana Datos.")
    else:
        data  = st.session_state["data"]
        media = st.session_state["media"]
        n     = st.session_state["n"]

        with st.expander("Referencia teorica — como funciona la prueba Z"):
            ref1, ref2 = st.columns(2)
            with ref1:
                st.markdown("""
**Hipotesis nula (H0):** Lo que asumimos cierto por defecto. Se pone a prueba.

**Hipotesis alternativa (H1):** Lo que buscamos demostrar con los datos.

**Estadistico Z:** Distancia entre la media muestral y el valor hipotetizado, medida en desviaciones estandar.

> Z = (x&#772; &#8722; &#956;&#8320;) / (&#963; / &#8730;n)
                """)
            with ref2:
                st.markdown("""
**p-value:** Probabilidad de obtener un Z tan extremo si H0 fuera cierta.
Si es muy pequeno, los datos son poco compatibles con H0.

**Nivel de significancia (alpha):** Umbral de decision.
Valores tipicos: 0.01, 0.05, 0.10.

**Regla:**
- p-value < alpha &rarr; Rechazar H0
- p-value &ge; alpha &rarr; No rechazar H0

**Requisito:** sigma poblacional conocida y n &ge; 30.
                """)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        st.markdown('<p class="slabel">Parametros</p>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            h0 = st.number_input("Valor hipotetizado (mu0)", value=50.0)
        with p2:
            sigma = st.number_input("Desv. estandar poblacional (sigma)", value=10.0, min_value=0.01)
        with p3:
            alpha = st.number_input("Significancia (alpha)", value=0.05,
                                    min_value=0.001, max_value=0.20, step=0.01)
        with p4:
            tipo_raw = st.selectbox("Tipo de prueba",
                                    ("Bilateral  —  mu distinto de mu0",
                                     "Cola izquierda  —  mu < mu0",
                                     "Cola derecha  —  mu > mu0"))

        if "izquierda" in tipo_raw:
            tipo   = "Cola izquierda"
            h1_str = f"mu < {h0}"
        elif "derecha" in tipo_raw:
            tipo   = "Cola derecha"
            h1_str = f"mu > {h0}"
        else:
            tipo   = "Bilateral"
            h1_str = f"mu distinto de {h0}"

        hd1, hd2 = st.columns(2)
        with hd1:
            st.markdown(f"""
<div class="card-sm" style="margin-top:14px;">
    <div class="slbl">Hipotesis nula</div>
    <div style="font-family:'DM Mono',monospace;font-size:1.05rem;color:var(--text);margin-top:8px;">H0 : mu = {h0}</div>
</div>""", unsafe_allow_html=True)
        with hd2:
            st.markdown(f"""
<div class="card-sm" style="margin-top:14px;">
    <div class="slbl">Hipotesis alternativa</div>
    <div style="font-family:'DM Mono',monospace;font-size:1.05rem;color:var(--accent);margin-top:8px;">H1 : {h1_str}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        z_stat = (media - h0) / (sigma / np.sqrt(n))

        if tipo == "Bilateral":
            p_value  = 2 * (1 - norm.cdf(abs(z_stat)))
            z_crit   = norm.ppf(1 - alpha / 2)
            reg_crit = f"|Z| > {z_crit:.4f}"
        elif tipo == "Cola derecha":
            p_value  = 1 - norm.cdf(z_stat)
            z_crit   = norm.ppf(1 - alpha)
            reg_crit = f"Z > {z_crit:.4f}"
        else:
            p_value  = norm.cdf(z_stat)
            z_crit   = norm.ppf(alpha)
            reg_crit = f"Z < {z_crit:.4f}"

        decision   = "RECHAZAR H0" if p_value < alpha else "NO RECHAZAR H0"
        rechazamos = p_value < alpha

        for k, v in [("z_stat", z_stat), ("p_value", p_value), ("z_crit", z_crit),
                     ("alpha", alpha), ("sigma", sigma), ("h0", h0),
                     ("tipo", tipo), ("decision", decision), ("reg_crit", reg_crit),
                     ("h1_str", h1_str)]:
            st.session_state[k] = v

        st.markdown('<p class="slabel">Resultados del calculo</p>', unsafe_allow_html=True)
        rc = st.columns(5)
        for col, lb, vl in zip(rc,
                                ["Z calculado", "Z critico", "p-value", "Region critica", "n"],
                                [f"{z_stat:.4f}", f"{z_crit:.4f}",
                                 f"{p_value:.6f}", reg_crit, str(n)]):
            with col:
                st.markdown(f"""
<div class="sblock">
    <div class="sval" style="font-size:1.05rem;">{vl}</div>
    <div class="slbl">{lb}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        vd1, vd2, vd3 = st.columns([1, 2, 1])
        with vd2:
            if rechazamos:
                st.markdown(f"""
<div class="verdict-r">
    <div class="vl">{decision}</div>
    <div class="vs">p-value {p_value:.6f} &lt; alpha {alpha} — evidencia estadistica suficiente</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="verdict-ok">
    <div class="vl">{decision}</div>
    <div class="vs">p-value {p_value:.6f} &ge; alpha {alpha} — evidencia insuficiente para rechazar H0</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if rechazamos:
            st.error(
                f"Con alpha = {alpha}, se rechaza H0. "
                f"Hay evidencia estadistica para afirmar que la media de la poblacion no es {h0}. "
                f"Resultado significativo al {(1-alpha)*100:.0f}% de confianza."
            )
        else:
            st.success(
                f"Con alpha = {alpha}, no se rechaza H0. "
                f"No hay evidencia suficiente para afirmar que la media difiere de {h0}. "
                f"Resultado no significativo al {(1-alpha)*100:.0f}% de confianza."
            )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Distribucion normal con region de rechazo</p>', unsafe_allow_html=True)
        st.markdown('<p class="note">La zona roja es la region de rechazo: si tu estadistico Z cae en ella, rechazamos H0. La zona azul es la region de no rechazo. La linea dorada es tu Z calculado.</p>', unsafe_allow_html=True)

        fig3, ax3 = new_fig(12, 5)
        xs = np.linspace(-4.4, 4.4, 700)
        ys = norm.pdf(xs)

        if tipo == "Bilateral":
            xl = xs[xs <= -z_crit]
            xr = xs[xs >= z_crit]
            xm = xs[(xs > -z_crit) & (xs < z_crit)]
            ax3.fill_between(xl, norm.pdf(xl), color=C_RED,  alpha=0.32)
            ax3.fill_between(xr, norm.pdf(xr), color=C_RED,  alpha=0.32,
                             label=f"Region de rechazo  alpha/2 = {alpha/2:.3f}")
            ax3.fill_between(xm, norm.pdf(xm), color=C_BLUE, alpha=0.10,
                             label=f"No rechazo  1-alpha = {1-alpha:.2f}")
            for zc in [-z_crit, z_crit]:
                ax3.axvline(zc, color=C_RED, lw=1.1, ls="--", alpha=0.65)
                ax3.text(zc, -0.017, f"Zc={zc:.2f}", ha="center",
                         color=C_RED, fontsize=8, fontfamily="monospace")
        elif tipo == "Cola derecha":
            xr = xs[xs >= z_crit]
            xl = xs[xs < z_crit]
            ax3.fill_between(xr, norm.pdf(xr), color=C_RED,  alpha=0.32,
                             label=f"Region de rechazo  alpha = {alpha}")
            ax3.fill_between(xl, norm.pdf(xl), color=C_BLUE, alpha=0.10,
                             label=f"No rechazo  1-alpha = {1-alpha:.2f}")
            ax3.axvline(z_crit, color=C_RED, lw=1.1, ls="--", alpha=0.65)
            ax3.text(z_crit, -0.017, f"Zc={z_crit:.2f}", ha="center",
                     color=C_RED, fontsize=8, fontfamily="monospace")
        else:
            xl = xs[xs <= z_crit]
            xr = xs[xs > z_crit]
            ax3.fill_between(xl, norm.pdf(xl), color=C_RED,  alpha=0.32,
                             label=f"Region de rechazo  alpha = {alpha}")
            ax3.fill_between(xr, norm.pdf(xr), color=C_BLUE, alpha=0.10,
                             label=f"No rechazo  1-alpha = {1-alpha:.2f}")
            ax3.axvline(z_crit, color=C_RED, lw=1.1, ls="--", alpha=0.65)
            ax3.text(z_crit, -0.017, f"Zc={z_crit:.2f}", ha="center",
                     color=C_RED, fontsize=8, fontfamily="monospace")

        ax3.plot(xs, ys, color=C_TEXT, lw=2, zorder=4, alpha=0.85)

        z_plot = float(np.clip(z_stat, -4.3, 4.3))
        ax3.axvline(z_plot, color=C_GOLD, lw=2.5, zorder=5,
                    label=f"Z observado = {z_stat:.4f}")
        ax3.scatter([z_plot], [norm.pdf(z_plot)], color=C_GOLD, s=80, zorder=6)

        ann_x  = z_plot + 0.85 if z_plot < 1.5 else z_plot - 0.85
        ann_ha = "left"        if z_plot < 1.5 else "right"
        ax3.annotate(
            f"Z = {z_stat:.4f}\np = {p_value:.4f}",
            xy=(z_plot, norm.pdf(z_plot)),
            xytext=(ann_x, norm.pdf(z_plot) + 0.065),
            ha=ann_ha, color=C_GOLD, fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=C_SURF,
                      edgecolor=C_GOLD, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.4),
        )

        ax3.set_xlabel("Estadistico Z estandar", fontsize=10)
        ax3.set_ylabel("Densidad de probabilidad", fontsize=10)
        ax3.set_title(
            f"Distribucion Normal  —  {tipo}  |  H0: mu = {h0}  |  alpha = {alpha}",
            fontsize=11, fontweight="bold", pad=14,
        )
        ax3.legend(facecolor=C_SURF, labelcolor=C_TEXT, fontsize=8.5,
                   framealpha=0.9, edgecolor=C_GRID, loc="upper right")
        ax3.set_xlim(-4.6, 4.6)
        ax3.set_ylim(-0.03, 0.47)
        st.pyplot(fig3, use_container_width=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Tabla resumen</p>', unsafe_allow_html=True)
        summ = pd.DataFrame({
            "Parametro": [
                "Media muestral (x-barra)", "Media hipotetizada (mu0)",
                "Desv. estandar poblacional (sigma)", "Tamano de muestra (n)",
                "Tipo de prueba", "Estadistico Z", "Z critico",
                "Region critica", "p-value", "Nivel de significancia (alpha)", "Decision",
            ],
            "Valor": [
                f"{media:.4f}", f"{h0}", f"{sigma}", f"{n}",
                tipo, f"{z_stat:.4f}", f"{z_crit:.4f}",
                reg_crit, f"{p_value:.6f}", f"{alpha}", decision,
            ],
        })
        st.dataframe(summ, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  TAB 4 — ASISTENTE IA
# ─────────────────────────────────────────────
def evaluar_analisis_local(val_sesgo, val_kurt, n_out, p_value, alpha, decision,
                            resp_normal, resp_sesgo, resp_outliers):
    """Evalua el analisis del estudiante localmente sin API."""
    puntos      = 0
    comentarios = []

    # 1. Normalidad
    es_normal_real = abs(val_sesgo) < 0.5 and abs(val_kurt) < 1
    dijo_normal    = "parece normal" in resp_normal.lower()
    dijo_no_normal = "no parece" in resp_normal.lower()
    dijo_seguro    = "no estoy seguro" in resp_normal.lower()

    if es_normal_real and dijo_normal:
        puntos += 3
        comentarios.append(("ok", "Normalidad",
            f"Correcto. Con sesgo = {val_sesgo:.3f} y curtosis = {val_kurt:.3f}, "
            f"la distribucion se aproxima a la normal (|sesgo| < 0.5 y |curtosis| < 1)."))
    elif not es_normal_real and dijo_no_normal:
        puntos += 3
        comentarios.append(("ok", "Normalidad",
            f"Correcto. Los valores de sesgo ({val_sesgo:.3f}) o curtosis ({val_kurt:.3f}) "
            f"indican alejamiento de la normalidad."))
    elif dijo_seguro:
        puntos += 1
        comentarios.append(("warn", "Normalidad",
            f"Respuesta neutral. Se esperaba una observacion mas definida. "
            f"El sesgo real es {val_sesgo:.3f} y la curtosis {val_kurt:.3f}."))
    elif resp_normal.lower() == "sin respuesta":
        puntos += 0
        comentarios.append(("err", "Normalidad",
            "No se proporciono respuesta sobre la normalidad."))
    else:
        comentarios.append(("err", "Normalidad",
            f"Incorrecto. El sesgo real es {val_sesgo:.3f} y la curtosis {val_kurt:.3f}. "
            f"{'La distribucion si es aproximadamente normal.' if es_normal_real else 'La distribucion no cumple criterios de normalidad.'}"))

    # 2. Sesgo
    sesgo_real_pos = val_sesgo >  0.5
    sesgo_real_neg = val_sesgo < -0.5
    sesgo_real_no  = abs(val_sesgo) <= 0.5

    if sesgo_real_no and "no hay sesgo" in resp_sesgo.lower():
        puntos += 3
        comentarios.append(("ok", "Sesgo",
            f"Correcto. El sesgo de {val_sesgo:.3f} esta dentro del rango simetrico (|sesgo| <= 0.5)."))
    elif sesgo_real_pos and "positivo" in resp_sesgo.lower():
        puntos += 3
        comentarios.append(("ok", "Sesgo",
            f"Correcto. El sesgo positivo de {val_sesgo:.3f} indica cola hacia la derecha."))
    elif sesgo_real_neg and "negativo" in resp_sesgo.lower():
        puntos += 3
        comentarios.append(("ok", "Sesgo",
            f"Correcto. El sesgo negativo de {val_sesgo:.3f} indica cola hacia la izquierda."))
    elif "no puedo" in resp_sesgo.lower():
        puntos += 1
        comentarios.append(("warn", "Sesgo",
            f"Respuesta evasiva. El sesgo real es {val_sesgo:.3f}; "
            f"es observable directamente en la asimetria del histograma."))
    elif resp_sesgo.lower() == "sin respuesta":
        puntos += 0
        comentarios.append(("err", "Sesgo",
            "No se proporciono respuesta sobre el sesgo."))
    else:
        comentarios.append(("err", "Sesgo",
            f"Incorrecto. El sesgo real es {val_sesgo:.3f}. "
            f"Observa hacia que lado se extiende la cola del histograma."))

    # 3. Outliers
    hay_outliers_real = n_out > 0
    dijo_si  = "si, hay" in resp_outliers.lower()
    dijo_no  = "no hay"  in resp_outliers.lower()
    dijo_pos = "posibles" in resp_outliers.lower()

    if hay_outliers_real and dijo_si:
        puntos += 2
        comentarios.append(("ok", "Outliers",
            f"Correcto. Se detectaron {n_out} valor(es) atipico(s) mediante el criterio IQR."))
    elif hay_outliers_real and dijo_pos:
        puntos += 1
        comentarios.append(("warn", "Outliers",
            f"Parcialmente correcto. Efectivamente hay {n_out} outlier(s) confirmados por IQR."))
    elif not hay_outliers_real and dijo_no:
        puntos += 2
        comentarios.append(("ok", "Outliers",
            "Correcto. No se detectaron valores atipicos fuera del rango IQR en el boxplot."))
    elif not hay_outliers_real and dijo_pos:
        puntos += 1
        comentarios.append(("warn", "Outliers",
            "Precaucion injustificada. No hay outliers confirmados por el criterio IQR."))
    elif resp_outliers.lower() == "sin respuesta":
        puntos += 0
        comentarios.append(("err", "Outliers",
            "No se proporciono respuesta sobre los outliers."))
    else:
        comentarios.append(("err", "Outliers",
            f"Incorrecto. El numero real de outliers es {n_out}. "
            f"Revisa los puntos fuera de los bigotes en el boxplot."))

    # 4. Decision prueba Z (siempre correcta porque la calcula el sistema)
    puntos += 2
    comentarios.append(("ok", "Decision Z",
        f"La decision '{decision}' es correcta segun p-value = {p_value:.6f} vs alpha = {alpha}. "
        f"{'Como p < alpha, se rechaza H0.' if p_value < alpha else 'Como p >= alpha, no se rechaza H0.'}"))

    calificacion = min(10, round(puntos))
    return comentarios, calificacion


with tab4:
    st.markdown('<p class="slabel">Paso 4</p><h2 class="stitle">Asistente de IA — Gemini</h2>', unsafe_allow_html=True)

    if st.session_state["data"] is None:
        st.warning("Primero carga tus datos y completa la prueba Z.")
    elif st.session_state["z_stat"] is None:
        st.warning("Primero ejecuta la prueba Z en la pestana correspondiente.")
    else:
        media     = st.session_state["media"]
        mediana   = st.session_state["mediana"]
        desv      = st.session_state["desv"]
        n         = st.session_state["n"]
        val_sesgo = st.session_state["val_sesgo"]
        val_kurt  = st.session_state["val_kurt"]
        n_out     = st.session_state["n_out"]
        z_stat    = st.session_state["z_stat"]
        p_value   = st.session_state["p_value"]
        z_crit    = st.session_state["z_crit"]
        alpha     = st.session_state["alpha"]
        sigma     = st.session_state["sigma"]
        h0        = st.session_state["h0"]
        tipo      = st.session_state["tipo"]
        decision  = st.session_state["decision"]
        reg_crit  = st.session_state["reg_crit"]
        h1_str    = st.session_state["h1_str"]

        resp_normal   = st.session_state.get("resp_normal")   or "Sin respuesta"
        resp_sesgo    = st.session_state.get("resp_sesgo")    or "Sin respuesta"
        resp_outliers = st.session_state.get("resp_outliers") or "Sin respuesta"
        resp_texto    = st.session_state.get("resp_texto")    or ""

        ai1, ai2 = st.columns([3, 2])

        with ai2:
            st.markdown(f"""
<div class="card">
    <div class="slabel">Resumen para la IA</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#8aafc8;line-height:2;margin-top:10px;">
        Media muestral &nbsp;&nbsp;&nbsp; {media:.4f}<br>
        H0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mu = {h0}<br>
        H1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {h1_str}<br>
        Tipo &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {tipo}<br>
        n &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {n}<br>
        sigma &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {sigma}<br>
        alpha &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {alpha}<br>
        Z calculado &nbsp;&nbsp;&nbsp;&nbsp; {z_stat:.4f}<br>
        p-value &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {p_value:.6f}<br>
        Decision &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {decision}
    </div>
    <div style="margin-top:18px;">
        <div class="slbl" style="margin-bottom:8px;">Tu analisis</div>
        <div style="font-size:0.8rem;color:#8aafc8;line-height:1.8;">
            Normal: {resp_normal}<br>
            Sesgo: {resp_sesgo}<br>
            Outliers: {resp_outliers}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

        with ai1:
            st.markdown('<p class="note">La IA recibe exclusivamente el resumen estadistico y tu analisis. No recibe los datos crudos.</p>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Consultar a Gemini"):
                analisis_est = (
                    f"Normalidad: {resp_normal} | "
                    f"Sesgo: {resp_sesgo} | "
                    f"Outliers: {resp_outliers}\n"
                    f"Explicacion: {resp_texto.strip() if resp_texto.strip() else 'No proporcionada'}"
                )

                prompt = f"""
Eres un profesor de estadistica evaluando a un estudiante de Ingenieria en Tecnologias de la Informacion.

RESUMEN ESTADISTICO:
- Media muestral: {media:.4f}
- Mediana: {mediana:.4f}
- Desviacion estandar muestral: {desv:.4f}
- Sigma poblacional usada: {sigma}
- Media hipotetizada H0: {h0}
- Hipotesis alternativa H1: {h1_str}
- n: {n}
- Alpha: {alpha}
- Tipo de prueba: {tipo}
- Z calculado: {z_stat:.4f}
- Z critico: {z_crit:.4f}
- Region critica: {reg_crit}
- p-value: {p_value:.6f}
- Decision automatica: {decision}
- Sesgo muestral: {val_sesgo:.4f}
- Curtosis: {val_kurt:.4f}
- Outliers detectados: {n_out}

ANALISIS DEL ESTUDIANTE:
{analisis_est}

TAREA: Responde en espanol, sin emojis, con parrafos bien separados, maximo 320 palabras:
1. Confirma o corrige la decision "{decision}" usando el p-value y el alpha.
2. Compara el analisis del estudiante (sesgo, normalidad, outliers) con los valores reales.
3. Un consejo educativo preciso sobre pruebas de hipotesis.
4. Calificacion del analisis del estudiante del 1 al 10 con justificacion breve.
"""

                with st.spinner("Procesando respuesta..."):
                    try:
                        resp_ia = gemini_model.generate_content(prompt)
                        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                        st.markdown('<p class="slabel">Retroalimentacion de Gemini</p>', unsafe_allow_html=True)
                        st.markdown(f"""
<div class="card">
    <div style="font-size:0.9rem;line-height:1.85;color:#c2d5ee;">
        {resp_ia.text.replace(chr(10), "<br>")}
    </div>
</div>""", unsafe_allow_html=True)

                    except Exception as exc:
                        # ── Mensaje de error explicativo ──
                        st.error(f"Error al conectar con la API de Gemini: {exc}")
                        st.markdown("""
<div class="card" style="border-color:rgba(251,191,36,0.35);background:rgba(251,191,36,0.05);margin-top:4px;">
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#fbbf24;
                letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
        Error 429 — Cuota de la API agotada
    </div>
    <div style="font-size:0.86rem;color:#c2d5ee;line-height:1.8;">
        La cuenta de Google Gemini utilizada ha alcanzado el limite de solicitudes
        gratuitas (error 429: Too Many Requests / Quota Exceeded). Esto no afecta
        el funcionamiento del resto de la aplicacion.<br><br>
        A continuacion se muestra una <strong style="color:#38bdf8;">evaluacion automatica local</strong>
        que analiza tu respuesta comparandola con los valores estadisticos reales.
    </div>
</div>""", unsafe_allow_html=True)

                        # ── Evaluacion local ──
                        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                        st.markdown('<p class="slabel">Evaluacion automatica local</p>', unsafe_allow_html=True)

                        comentarios, calificacion = evaluar_analisis_local(
                            val_sesgo, val_kurt, n_out, p_value, alpha, decision,
                            resp_normal, resp_sesgo, resp_outliers
                        )

                        ICONO_MAP = {
                            "ok":   ("✓", "#34d399", "rgba(52,211,153,0.28)",  "rgba(52,211,153,0.05)"),
                            "warn": ("~", "#fbbf24", "rgba(251,191,36,0.28)",  "rgba(251,191,36,0.05)"),
                            "err":  ("✗", "#f87171", "rgba(248,113,113,0.28)", "rgba(248,113,113,0.05)"),
                        }

                        for tipo_icono, categoria, mensaje in comentarios:
                            icono, color, borde, bg = ICONO_MAP[tipo_icono]
                            st.markdown(f"""
<div class="card-sm" style="border-color:{borde};background:{bg};margin-bottom:10px;">
    <div style="display:flex;align-items:flex-start;gap:14px;">
        <div style="font-size:1.15rem;color:{color};margin-top:2px;font-weight:700;">{icono}</div>
        <div>
            <div style="font-family:'DM Mono',monospace;font-size:0.69rem;color:{color};
                        letter-spacing:0.09em;text-transform:uppercase;margin-bottom:5px;">
                {categoria}
            </div>
            <div style="font-size:0.84rem;color:#c2d5ee;line-height:1.65;">{mensaje}</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

                        # ── Calificacion final ──
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_cal = st.columns([1, 2, 1])
                        with col_cal[1]:
                            if calificacion >= 8:
                                color_cal = "#34d399"
                                nivel     = "Excelente"
                            elif calificacion >= 6:
                                color_cal = "#fbbf24"
                                nivel     = "Aceptable"
                            else:
                                color_cal = "#f87171"
                                nivel     = "Necesita mejorar"

                            st.markdown(f"""
<div class="sblock" style="padding:30px 20px;">
    <div class="sval" style="font-size:2.6rem;color:{color_cal};">{calificacion} / 10</div>
    <div class="slbl" style="margin-top:10px;">Calificacion del analisis</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{color_cal};
                margin-top:8px;letter-spacing:0.08em;text-transform:uppercase;">
        {nivel}
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="pgfooter">
    Probabilidad y Estadistica &nbsp;&nbsp;·&nbsp;&nbsp; Ingenieria en Tecnologias de la Informacion<br>
    Streamlit &nbsp;·&nbsp; Google Gemini 2.0 &nbsp;·&nbsp; SciPy &nbsp;·&nbsp; Seaborn
</div>
""", unsafe_allow_html=True)