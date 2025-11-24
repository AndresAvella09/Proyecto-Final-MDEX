import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px
import streamlit as st
import io

# Configuración de la página
st.set_page_config(
    page_title="Análisis EDA - Aguacate",
    page_icon="🥑",
    layout="wide"
)

# Título de la aplicación
st.title("Análisis Exploratorio - Datos de Aguacate")
st.markdown("---")

# Función para cargar datos desde bytes (no contiene widgets)
@st.cache_data
def load_data(file_bytes):
    if file_bytes is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        st.error("Error al parsear el CSV. Verifica el formato del archivo.")
        return None


# Widget para subir archivo (fuera de la función cacheada)
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
file_bytes = uploaded_file.read() if uploaded_file is not None else None

# Cargar datos
df = load_data(file_bytes)

if df is not None:
    # Sidebar para controles
    st.sidebar.header("Controles de Análisis")

    # Selección de tratamiento para filtrar
    tratamientos = df['Tratamiento'].unique()
    tratamiento_seleccionado = st.sidebar.multiselect(
        "Filtrar por Tratamiento:",
        options=tratamientos,
        default=tratamientos
    )

    # Filtrar datos
    df_filtrado = df[df['Tratamiento'].isin(tratamiento_seleccionado)]

    # ===== SECCIÓN 1: DIAGNÓSTICO INICIAL =====
    st.header("📊 Diagnóstico Inicial del Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dimensiones", f"{df_filtrado.shape[0]} filas × {df_filtrado.shape[1]} columnas")

    with col2:
        nulos_total = df_filtrado.isnull().sum().sum()
        st.metric("Valores nulos totales", nulos_total)

    with col3:
        st.metric("Tratamientos únicos", len(df_filtrado['Tratamiento'].unique()))

    # Expanders para información detallada
    with st.expander("🔍 Valores nulos por columna"):
        nulos = df_filtrado.isnull().sum()
        st.dataframe(nulos, use_container_width=True)

    with st.expander("📝 Tipos de datos"):
        tipos = df_filtrado.dtypes
        st.dataframe(tipos, use_container_width=True)

    with st.expander("📈 Estadísticas descriptivas del Peso"):
        st.dataframe(df_filtrado['Peso_g'].describe(), use_container_width=True)

    # ===== SECCIÓN 2: BALANCEAMIENTO =====
    st.header("⚖️ Balanceamiento del Experimento")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Observaciones por tratamiento")
        conteo_tratamientos = df_filtrado['Tratamiento'].value_counts()
        st.bar_chart(conteo_tratamientos)
        st.dataframe(conteo_tratamientos, use_container_width=True)

    with col2:
        st.subheader("Árboles únicos por tratamiento")
        arboles_por_tratamiento = df_filtrado.groupby('Tratamiento')['Arbol_ID'].nunique()
        st.bar_chart(arboles_por_tratamiento)
        st.dataframe(arboles_por_tratamiento, use_container_width=True)

    # ===== SECCIÓN 3: ANÁLISIS DE VARIANZA =====
    st.header("📊 Análisis de Varianza")

    def analizar_varianza_jerarquica(df_analisis):
        var_total = df_analisis['Peso_g'].var()
        promedios_arboles = df_analisis.groupby('Arbol_ID')['Peso_g'].mean()
        var_entre_arboles = promedios_arboles.var()
        var_dentro_arboles = df_analisis.groupby('Arbol_ID')['Peso_g'].var().mean()
        promedios_tratamientos = df_analisis.groupby('Tratamiento')['Peso_g'].mean()
        var_entre_tratamientos = promedios_tratamientos.var()

        resultados = {
            'total': var_total,
            'entre_arboles': var_entre_arboles,
            'dentro_arboles': var_dentro_arboles,
            'entre_tratamientos': var_entre_tratamientos
        }

        # Mostrar resultados en Streamlit
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Varianza Total", f"{var_total:.2f}")

        with col2:
            st.metric("Entre Árboles", f"{var_entre_arboles:.2f}", 
                     f"{(var_entre_arboles/var_total*100):.1f}%")

        with col3:
            st.metric("Dentro Árboles", f"{var_dentro_arboles:.2f}", 
                     f"{(var_dentro_arboles/var_total*100):.1f}%")

        with col4:
            st.metric("Entre Tratamientos", f"{var_entre_tratamientos:.2f}", 
                     f"{(var_entre_tratamientos/var_total*100):.1f}%")

        return resultados

    componentes = analizar_varianza_jerarquica(df_filtrado)

    # ===== SECCIÓN 4: VISUALIZACIONES =====
    st.header("📈 Visualizaciones")

    # Tabs para diferentes tipos de gráficas
    tab1, tab2, tab3, tab4 = st.tabs([
        "Boxplot", "Histogramas", "Strip Plot", "Violin Plot"
    ])

    with tab1:
        st.subheader("Distribución del Peso por Tratamiento")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Tratamiento', y='Peso_g', data=df_filtrado, ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Histogramas por Tratamiento")
        # Crear subplots para histogramas
        tratamientos_unicos = df_filtrado['Tratamiento'].unique()
        n_tratamientos = len(tratamientos_unicos)
        cols = 2
        rows = (n_tratamientos + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        axes = axes.flatten()

        for i, tratamiento in enumerate(tratamientos_unicos):
            if i < len(axes):
                datos_tratamiento = df_filtrado[df_filtrado['Tratamiento'] == tratamiento]['Peso_g']
                axes[i].hist(datos_tratamiento, bins=15, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Tratamiento: {tratamiento}')
                axes[i].set_xlabel('Peso (g)')
                axes[i].set_ylabel('Frecuencia')

        # Ocultar ejes vacíos
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Strip Plot por Tratamiento")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.stripplot(x="Tratamiento", y="Peso_g", data=df_filtrado, 
                     jitter=True, size=4, ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("Violin Plot Interactivo")
        fig = px.violin(df_filtrado, x="Tratamiento", y="Peso_g", 
                       color="Tratamiento", box=True, points="all",
                       title="Distribución de Peso por Tratamiento")
        st.plotly_chart(fig, use_container_width=True)

    # ===== SECCIÓN 5: ESTADÍSTICAS POR TRATAMIENTO =====
    st.header("📋 Estadísticas Detalladas por Tratamiento")

    stats_tratamiento = df_filtrado.groupby('Tratamiento')['Peso_g'].describe()
    st.dataframe(stats_tratamiento, use_container_width=True)

    # ===== SECCIÓN 6: DATOS CRUDOS =====
    with st.expander("🔢 Ver Datos Crudos"):
        st.dataframe(df_filtrado, use_container_width=True)

        # Opción para descargar datos filtrados
        csv = df_filtrado.to_csv(index=False)
        st.download_button(
            label="📥 Descargar datos filtrados como CSV",
            data=csv,
            file_name="datos_aguacate_filtrados.csv",
            mime="text/csv"
        )

else:
    # Mensaje inicial cuando no hay datos
    st.markdown("""
    ## Bienvenido al Análisis Exploratorio de Datos de Aguacate
    
    **Para comenzar:**
    1. Sube tu archivo CSV usando el uploader arriba
    2. Explora las diferentes secciones del análisis
    3. Usa los controles en la barra lateral para filtrar datos
    
    **El análisis incluye:**
    - Diagnóstico inicial de datos
    - Análisis de balanceamiento experimental  
    - Descomposición de varianza
    - Múltiples visualizaciones
    - Estadísticas por tratamiento
    """)

# Footer
st.markdown("---")
st.markdown("*Análisis EDA - Datos de Aguacate*")