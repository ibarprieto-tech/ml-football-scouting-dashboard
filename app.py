import streamlit as st
import pandas as pd
import plotly.express as px
import re
import collections

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Dashboard - ML & Football Scouting",
    page_icon="⚽",
    layout="wide"
)

# Forzar estilo claro en Plotly
px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = ["#1D4ED8"]  # azul fuerte

# ==========================================
# ESTILOS
# ==========================================
st.markdown("""
    <style>

    /* ======== TÍTULO PRINCIPAL ======== */
    .main-title { 
        font-size:30px !important; 
        font-weight: bold; 
        color: #FFFFFF !important; 
        text-align: center; 
        margin-bottom: 10px; 
    }

    /* ======== SUBTÍTULO ======== */
    .subtitle { 
        font-size:16px !important; 
        color: #FFFFFF !important; 
        text-align: center; 
        margin-bottom: 30px; 
    }

    /* ======== AJUSTES DEL SIDEBAR ======== */

    /* Cambiar tamaño de fuente del sidebar */
    [data-testid="stSidebar"] * {
        font-size: 15px !important;     /
        color: white !important;        /
    }

    /* Cambiar ancho del sidebar */
    [data-testid="stSidebar"] {
        width: 400px !important;        /
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 400px !important;        /
    }

    /* Reducir espacio vertical entre elementos del sidebar */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: -10px !important;   /
    }

    </style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("🔍 Información del Proyecto")

    st.markdown("**Pregunta de Investigación:**")
    st.info(
        "¿De qué manera el uso de algoritmos supervisados de machine learning "
        "optimiza la precisión del scouting para la identificación de talento "
        "emergente en el fútbol profesional?"
    )
    st.markdown("**Keywords Autorizadas:**")
    st.code(
        '1. "Machine learning"\n'
        '2. "Scouting"\n'
        '3. "Talent identification"\n'
        '4. "Soccer"'
    )

    st.markdown("---")
    st.subheader("📥 Carga de Datos")
    uploaded_file = st.file_uploader("Sube el archivo CSV", type=["csv"])

# ==========================================
# CARGA DE DATOS
# ==========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = None
if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("Dataset cargado con éxito")

# ==========================================
# TÍTULOS PRINCIPALES (COLOR BLANCO)
# ==========================================
st.markdown(
    '<div class="main-title">⚽ Machine Learning y Búsqueda de Talento</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Lectura y análisis del archivo</div>',
    unsafe_allow_html=True
)

# ==========================================
# SI HAY DATA → MOSTRAR DASHBOARD
# ==========================================
if df is not None:

    required_cols = ["Authors", "Title", "Year", "Abstract", "Cited by", "Source title"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"❌ El archivo no contiene las columnas necesarias: {missing}")
        st.stop()
    else:
        st.success("✅ El dataset contiene todos los metadatos esenciales requeridos por Scopus.")

    # ==========================================
    # FILTROS
    # ==========================================
    st.markdown("### 🎛️ Filtros Globales")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        years = sorted(df["Year"].dropna().unique())
        selected_years = st.multiselect("Filtrar por Año:", years, default=years)

    with col_f2:
        min_cit, max_cit = int(df["Cited by"].min()), int(df["Cited by"].max())
        cit_range = st.slider("Rango de Citaciones:", min_cit, max_cit, (min_cit, max_cit))

    df_filtered = df[
        (df["Year"].isin(selected_years)) &
        (df["Cited by"].between(cit_range[0], cit_range[1]))
    ]

    # ==========================================
    # MÉTRICAS
    # ==========================================
    st.markdown("### 📊 Resumen General")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Artículos", len(df_filtered))
    col2.metric("Años", f"{df_filtered['Year'].min()} - {df_filtered['Year'].max()}")
    col3.metric("Citaciones Totales", int(df_filtered['Cited by'].sum()))
    col4.metric("Máx Citaciones", int(df_filtered['Cited by'].max()))

    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📋 Explorador", "📊 Gráficos", "🔤 Análisis de Texto"])

    # ==========================================
    # TAB 1
    # ==========================================
    with tab1:
        st.subheader("📋 Explorador del Dataset")
        st.dataframe(df_filtered, use_container_width=True)

    # ==========================================
    # FUNCIÓN PARA APLICAR ESTILO A TODOS LOS GRÁFICOS
    # ==========================================
    def apply_style(fig):
        fig.update_layout(
            font=dict(color="#000000", size=16),
            title_font=dict(color="#000000", size=22),
            paper_bgcolor="white",
            plot_bgcolor="white",

            xaxis=dict(
                tickfont=dict(color="#000000", size=14),
                title=dict(font=dict(color="#000000", size=16)),
                linecolor="#000000",
                gridcolor="#D1D5DB"
            ),

            yaxis=dict(
                tickfont=dict(color="#000000", size=14),
                title=dict(font=dict(color="#000000", size=16)),
                linecolor="#000000",
                gridcolor="#D1D5DB"
            )
        )
        return fig

    # ==========================================
    # TAB 2 — GRÁFICOS
    # ==========================================
    with tab2:
        st.subheader("📊 Visualizaciones Interactivas")

        # ============================
        # 1. Publicaciones por Año
        # ============================
        st.markdown("#### 📈 Publicaciones por Año")
        year_counts = df_filtered["Year"].value_counts().sort_index()

        fig_years = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={"x": "Año", "y": "Cantidad de Artículos"},
            title="Publicaciones por Año"
        )
        fig_years.update_traces(marker_color="#1D4ED8")
        st.plotly_chart(apply_style(fig_years), use_container_width=True)

        # ============================
        # 2. Top 5 Citados (NO MODIFICAR)
        # ============================
        st.markdown("#### 🏆 Top 5 Artículos más Citados")

        top5 = df_filtered.sort_values("Cited by", ascending=False).head(5).copy()
        top5["Short Title"] = top5["Title"].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)

        fig_top = px.bar(
            top5.sort_values("Cited by"),
            x="Cited by",
            y="Short Title",
            orientation="h",
            title="Top 5 Artículos más Citados"
        )
        fig_top.update_traces(marker_color="#1E40AF")
        st.plotly_chart(apply_style(fig_top), use_container_width=True)

        # ============================
        # 3. Distribución de Citaciones (MEJORADO)
        # ============================
        st.markdown("#### 📊 Distribución de Citaciones")

        fig_hist = px.histogram(
            df_filtered,
            x="Cited by",
            nbins=10,
            title="Distribución de Citaciones"
        )
        fig_hist.update_traces(marker_color="#1E3A8A")
        st.plotly_chart(apply_style(fig_hist), use_container_width=True)

        # ==========================================
        # NUEVA SECCIÓN: ARTÍCULOS POR RANGO
        # ==========================================
        st.markdown("### 📄 Artículos por Rango de Citaciones")

        bins = [0, 10, 20, 40, 60, 80, 100]
        labels = ["0–10", "10–20", "20–40", "40–60", "60–80", "80–100"]

        df_filtered["cit_range"] = pd.cut(
            df_filtered["Cited by"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        for label in labels:
            subset = df_filtered[df_filtered["cit_range"] == label]

            if len(subset) > 0:
                st.markdown(f"#### Rango {label} citaciones ({len(subset)} artículos)")
                st.dataframe(
                    subset[["Title", "Cited by", "Year", "Source title"]],
                    use_container_width=True
                )

    # ==========================================
    # TAB 3 — ANÁLISIS DE TEXTO
    # ==========================================
    with tab3:
        st.subheader("🔤 Frecuencia de Palabras en Abstracts")

        stopwords = set([
            'the','a','in','of','and','to','is','for','with','on','by','at','an','this','that','from','as','are','it','we',
            'player','players','football','soccer','scouting','talent','identification','machine','learning','data','using',
            'used','analysis','team','sports','study','results','proposed','performance','our','based','was','were','their',
            'how','which'
        ])

        words_list = []
        for abstract in df_filtered["Abstract"].dropna():
            words = re.findall(r'\b\w+\b', abstract.lower())
            words_list.extend([w for w in words if w not in stopwords and len(w) > 2])

        word_counts = collections.Counter(words_list).most_common(15)

        if word_counts:
            words_df = pd.DataFrame(word_counts, columns=["Palabra", "Frecuencia"])

            fig3 = px.bar(
                words_df.sort_values("Frecuencia"),
                x="Frecuencia",
                y="Palabra",
                title="Palabras Más Frecuentes en Abstracts"
            )

            fig3.update_traces(marker_color="#F97316")
            st.plotly_chart(apply_style(fig3), use_container_width=True)

        else:
            st.info("No se encontraron suficientes palabras para el análisis.")

else:
    st.info("💡 Sube un archivo CSV desde el menú lateral para comenzar.")
