import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import re

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA (Criterio 6: Empatía)
# ==========================================
st.set_page_config(
    page_title="Dashboard - ML & Football Scouting",
    page_icon="⚽",
    layout="wide"
)

# Estilo personalizado con CSS embebido
st.markdown("""
    <style>
    .main-title { font-size:40px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 20px; }
    .subtitle { font-size:18px !important; color: #4B5563; text-align: center; margin-bottom: 30px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- PANEL LATERAL DE NAVEGACIÓN E INFORMACIÓN ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5323/5323871.png", width=100)
    st.header("🔍 Información del Proyecto")
    st.markdown("**Pregunta de Investigación:**")
    st.info("¿De qué manera el uso de algoritmos supervisados de machine learning optimiza la precisión del scouting para la identificación de talento emergente en el fútbol profesional?")
    
    st.markdown("**Keywords Autorizadas (Scopus):**")
    st.code('1. "Machine learning"\n2. "Scouting"\n3. "Talent identification"\n4. "Soccer"')
    
    st.markdown("---")
    st.subheader("📥 Carga de Datos")
    # Widget de carga dinámica (Criterio 3: Funcionalidad)
    uploaded_file = st.file_uploader("Sube el archivo Scopus CSV exportado", type=["csv"])

# --- CARGA OPTIMIZADA Y CACHÉ DE DATOS ---
@st.cache_data
def load_data(file_source):
    df = pd.read_csv(file_source)
    return df

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ ¡Dataset local cargado con éxito!")
else:
    # Intento de consumo automatizado desde GitHub público (Criterio 3)
    try:
        # Reemplaza esta URL con tu enlace raw real de GitHub cuando crees tu repositorio
        github_url = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPOSITORIO/main/scopus_export.csv"
        df = pd.read_csv(github_url)
        st.sidebar.info("📊 Consumiendo dataset directamente desde repositorio de GitHub.")
    except:
        st.sidebar.warning("⚠️ Esperando carga de archivo CSV local o configuración de GitHub.")

# --- CUERPO PRINCIPAL DEL DASHBOARD ---
st.markdown('<div class="main-title">⚽ Machine Learning & Scouting de Fútbol</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis del Estado del Arte de la Literatura Científica (Scopus)</div>', unsafe_allow_html=True)

if df is not None:
    # Métricas de Resumen de Alto Nivel
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Artículos", len(df))
    with col_m2:
        st.metric("Rango de Años", f"{df['Year'].min()} - {df['Year'].max()}")
    with col_m3:
        st.metric("Total de Citaciones", int(df['Cited by'].sum()))
    with col_m4:
        st.metric("Máx Citaciones en un Paper", int(df['Cited by'].max()))

    # Organización de Pestañas (Criterio 4 y 6: Intuitivo y Ordenado)
    tab1, tab2, tab3 = st.tabs(["📋 Explorador del Dataset", "📊 Gráficos Estadísticos", "🔤 Análisis de Texto (Abstracts)"])

    # PESTAÑA 1: EXPLORADOR DEL DATASET
    with tab1:
        st.subheader("Artículos Científicos Identificados")
        st.markdown("Utiliza la tabla interactiva de abajo para explorar, ordenar y filtrar los metadatos esenciales del dataset extraído.")
        
        # Filtro interactivo por Año
        years = sorted(df['Year'].unique())
        selected_years = st.multiselect("Filtrar por año de publicación:", years, default=years)
        df_filtered = df[df['Year'].isin(selected_years)]
        
        columns_to_show = ['Authors', 'Title', 'Year', 'Source title', 'Cited by', 'DOI']
        st.dataframe(df_filtered[columns_to_show], use_container_width=True)

    # PESTAÑA 2: GRÁFICOS ESTADÍSTICOS (Criterio 4: Calidad de Visualización)
    with tab2:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("📈 Publicaciones por Año")
            st.markdown("*Muestra la tendencia y el crecimiento de la investigación en este campo específico.*")
            year_counts = df['Year'].value_counts().sort_index()
            
            fig1, ax1 = plt.subplots(figsize=(7, 4.5))
            sns.barplot(x=year_counts.index, y=year_counts.values, hue=year_counts.index, palette="Blues_d", legend=False, ax=ax1)
            ax1.set_xlabel("Año")
            ax1.set_ylabel("Cantidad de Artículos")
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig1)

        with col_g2:
            st.subheader("🏆 Top 5 Artículos más Citados")
            st.markdown("*Identifica las publicaciones de mayor impacto y referencia para nuestra pregunta.*")
            top_cited = df[['Title', 'Cited by']].sort_values(by='Cited by', ascending=False).head(5)
            # Ordenar barras de menor a mayor para que la más grande quede arriba en el gráfico horizontal
            top_cited = top_cited.sort_values(by='Cited by', ascending=True)
            top_cited['Short Title'] = top_cited['Title'].apply(lambda x: x[:35] + '...' if len(x) > 35 else x)
            
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            sns.barplot(x='Cited by', y='Short Title', data=top_cited, hue='Short Title', palette="viridis", legend=False, ax=ax2)
            ax2.set_xlabel("Número de Citaciones")
            ax2.set_ylabel("Título del Artículo")
            st.pyplot(fig2)

    # PESTAÑA 3: ANÁLISIS DE CONTENIDO (Criterio 4)
    with tab3:
        st.subheader("🔤 Frecuencia de Conceptos Clave en los Resúmenes")
        st.markdown("Este gráfico muestra las palabras con mayor frecuencia dentro de los *Abstracts*, excluyendo conectores comunes. Permite mapear los enfoques metodológicos abordados.")
        
        # Procesamiento básico de texto libre de librerías pesadas
        words_list = []
        stopwords = set(['the', 'a', 'in', 'of', 'and', 'to', 'is', 'for', 'with', 'on', 'by', 'at', 'an', 'this', 'that', 'from', 'as', 'are', 'it', 'we', 'player', 'players', 'football', 'soccer', 'scouting', 'talent', 'identification', 'machine', 'learning', 'data', 'using', 'used', 'analysis', 'team', 'sports', 'study', 'results', 'proposed', 'performance', 'our', 'based', 'was', 'were', 'their', 'how', 'which'])
        
        for abstract in df['Abstract'].dropna():
            words = re.findall(r'\b\w+\b', abstract.lower())
            for w in words:
                if w not in stopwords and len(w) > 2:
                    words_list.append(w)
                    
        word_counts = collections.Counter(words_list).most_common(10)
        
        if word_counts:
            words_df = pd.DataFrame(word_counts, columns=['Palabra', 'Frecuencia']).sort_values(by='Frecuencia', ascending=True)
            
            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            sns.barplot(x='Frecuencia', y='Palabra', data=words_df, hue='Palabra', palette="rocket", legend=False, ax=ax3)
            ax3.set_xlabel("Frecuencia de la palabra")
            ax3.set_ylabel("Palabra Clave")
            st.pyplot(fig3)
        else:
            st.info("No se encontraron suficientes palabras para el análisis.")

else:
    # Estado inicial amigable (Criterio 6)
    st.info("💡 Consejo para iniciar: Sube el archivo CSV de Scopus desde el menú lateral para activar las pestañas y visualizaciones interactivas.")