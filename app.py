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

# Configuración de la página
st.set_page_config(
    page_title="Análisis EDA - Aguacate",
    page_icon="🥑",
    layout="wide"
)

# Título de la aplicación
st.title("Análisis Exploratorio - Datos de Aguacate")
st.markdown("---")

# Widget para subir archivo (el usuario sube directamente el CSV)
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

# Leer el CSV sólo cuando el usuario lo sube
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Error al leer el CSV. Verifica el formato y la codificación del archivo.")
        df = None
else:
    df = None

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
        "Violin Plot", "Histogramas", "Strip Plot", "Boxplot"
    ])

    with tab1:
        st.subheader("Violin Plot Interactivo")
        fig = px.violin(df_filtrado, x="Tratamiento", y="Peso_g", 
                       color="Tratamiento", box=True, points="all",
                       title="Distribución de Peso por Tratamiento")
        st.plotly_chart(fig, use_container_width=True)

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
        st.subheader("Distribución del Peso por Tratamiento")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Tratamiento', y='Peso_g', data=df_filtrado, ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)


    # ===== SECCIÓN 5: ESTADÍSTICAS POR TRATAMIENTO =====
    st.header("📋 Estadísticas Detalladas por Tratamiento")

    stats_tratamiento = df_filtrado.groupby('Tratamiento')['Peso_g'].describe()
    st.dataframe(stats_tratamiento, use_container_width=True)

    # ===== SECCIÓN 6: INFERENCIA =====

    # Función para formatear valores p
    def format_p_value(p_value):
        if p_value < 0.0001:
            return f"{p_value:.2e}"
        elif p_value < 0.001:
            return f"{p_value:.6f}"
        else:
            return f"{p_value:.4f}"


    st.header("📈 Análisis Estadístico Inferencial")

    # Verificar que hay suficientes datos para ANOVA
    if len(df_filtrado['Tratamiento'].unique()) < 2:
        st.warning("Se necesitan al menos 2 tratamientos para realizar ANOVA")
    else:
        # Definir nivel_significancia aquí para asegurar que esté disponible
        nivel_significancia = st.sidebar.slider(
            "Nivel de significancia (α):",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="Nivel de significancia para las pruebas estadísticas"
        )
        
        # ANOVA
        st.subheader("🔬 ANOVA - Análisis de Varianza")

        with st.spinner("Calculando ANOVA..."):
            try:
                # ANOVA table
                modelo = ols('Peso_g ~ C(Tratamiento)', data=df_filtrado).fit()
                anova_tabla = sm.stats.anova_lm(modelo, typ=2)
                
                # Guardar todos los resultados del ANOVA en variables
                suma_cuadrados_tratamiento = anova_tabla['sum_sq']['C(Tratamiento)']
                grados_libertad_tratamiento = anova_tabla['df']['C(Tratamiento)']
                
                # Calcular cuadrados medios manualmente si no están en la tabla
                if 'mean_sq' in anova_tabla.columns:
                    cuadrado_medio_tratamiento = anova_tabla['mean_sq']['C(Tratamiento)']
                    cuadrado_medio_residual = anova_tabla['mean_sq']['Residual']
                else:
                    # Calcular manualmente: CM = SC / gl
                    cuadrado_medio_tratamiento = suma_cuadrados_tratamiento / grados_libertad_tratamiento
                    suma_cuadrados_residual = anova_tabla['sum_sq']['Residual']
                    grados_libertad_residual = anova_tabla['df']['Residual']
                    cuadrado_medio_residual = suma_cuadrados_residual / grados_libertad_residual
                
                valor_F = anova_tabla['F']['C(Tratamiento)']
                p_valor = anova_tabla['PR(>F)']['C(Tratamiento)']
                
                # Asegurarnos de tener estos valores definidos
                if 'suma_cuadrados_residual' not in locals():
                    suma_cuadrados_residual = anova_tabla['sum_sq']['Residual']
                if 'grados_libertad_residual' not in locals():
                    grados_libertad_residual = anova_tabla['df']['Residual']
                
                # Mostrar tabla ANOVA
                st.write("**Tabla ANOVA:**")
                
                # Crear una copia de la tabla para formatear
                anova_formateada = anova_tabla.copy()
                
                # Si no existe la columna mean_sq, agregarla
                if 'mean_sq' not in anova_formateada.columns:
                    anova_formateada['mean_sq'] = anova_formateada['sum_sq'] / anova_formateada['df']
                
                # Formatear la columna de valor p
                if 'PR(>F)' in anova_formateada.columns:
                    anova_formateada['PR(>F)'] = anova_formateada['PR(>F)'].apply(format_p_value)
                
                st.dataframe(anova_formateada.style.format({
                    'sum_sq': '{:.4f}',
                    'df': '{:.0f}',
                    'mean_sq': '{:.4f}',
                    'F': '{:.4f}'
                }), use_container_width=True)
                
                # Mostrar resultados clave del ANOVA
                st.write("**Resultados clave del ANOVA:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Valor F", f"{valor_F:.4f}")
                
                with col2:
                    p_valor_formateado = format_p_value(p_valor)
                    st.metric("Valor p", p_valor_formateado)
                
                with col3:
                    st.metric("G.L. Tratamiento", f"{grados_libertad_tratamiento:.0f}")
                
                with col4:
                    st.metric("G.L. Residual", f"{grados_libertad_residual:.0f}")
                
                # Interpretación del resultado ANOVA
                if p_valor < nivel_significancia:
                    st.success(f"✅ **Resultado significativo**: Existen diferencias significativas entre los tratamientos (p < {nivel_significancia})")
                else:
                    st.info(f"🔍 **Resultado no significativo**: No hay evidencia suficiente de diferencias entre tratamientos (p ≥ {nivel_significancia})")
                
            except Exception as e:
                st.error(f"Error en el cálculo de ANOVA: {str(e)}")
                
        # Pruebas post-hoc Tukey HSD
        st.subheader("📊 Comparaciones Múltiples (Tukey HSD)")
        
        with st.spinner("Realizando comparaciones post-hoc..."):
            try:
                tukey_resultado = pairwise_tukeyhsd(
                    endog=df_filtrado['Peso_g'], 
                    groups=df_filtrado['Tratamiento'], 
                    alpha=nivel_significancia
                )
                
                # Mostrar resultados en tabla
                st.write("**Resultados de Tukey HSD:**")
                
                # Convertir a DataFrame para mejor visualización
                tukey_df = pd.DataFrame(
                    data=tukey_resultado._results_table.data[1:],
                    columns=tukey_resultado._results_table.data[0]
                )
                
                # Formatear valores p en notación científica si son muy pequeños
                tukey_df['p-adj'] = tukey_df['p-adj'].apply(lambda x: format_p_value(x))
                
                st.dataframe(tukey_df.style.format({
                    'group1': '{}',
                    'group2': '{}',
                    'meandiff': '{:.4f}',
                    'lower': '{:.4f}',
                    'upper': '{:.4f}',
                    'reject': '{}'
                }), use_container_width=True)
                
                # Visualización de los resultados de Tukey
                st.write("**Gráfico de comparaciones de Tukey:**")
                fig_tukey, ax = plt.subplots(figsize=(10, 6))
                tukey_resultado.plot_simultaneous(ax=ax)
                plt.title('Comparaciones Múltiples - Tukey HSD')
                plt.tight_layout()
                st.pyplot(fig_tukey)
                
                # Resumen de comparaciones significativas
                st.write("**Resumen de comparaciones significativas:**")
                # Convertir 'reject' a booleano para la comparación
                comparaciones_sig = tukey_df[tukey_df['reject'].astype(bool)]
                if len(comparaciones_sig) > 0:
                    for _, row in comparaciones_sig.iterrows():
                        st.write(f"- **{row['group1']} vs {row['group2']}**: "
                            f"Diferencia = {row['meandiff']:.2f}g, "
                            f"p = {row['p-adj']}")
                else:
                    st.write("No se encontraron diferencias significativas entre los pares de tratamientos.")
                    
            except Exception as e:
                st.error(f"Error en el cálculo de Tukey HSD: {str(e)}")

    # ===== SECCIÓN 7: DATOS CRUDOS =====
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