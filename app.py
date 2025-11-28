import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go



# Configuración de la página
st.set_page_config(
    page_title="Análisis EDA - Aguacate",
    layout="wide"
)


st.set_page_config(
    page_title="Análisis EDA - Aguacate",
    layout="wide"
)

# CSS personalizado para el color de fondo - ESTO VA INMEDIATAMENTE DESPUÉS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e9ecef;
    }
    
    /* Opcional: mejorar contraste para otros elementos */
    .main .block-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Opcional: ajustar sidebar */
    .css-1d391kg {
        background-color: #e9edc9;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    st.header("Diagnóstico Inicial del Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dimensiones", f"{df_filtrado.shape[0]} filas × {df_filtrado.shape[1]} columnas")

    with col2:
        nulos_total = df_filtrado.isnull().sum().sum()
        st.metric("Valores nulos totales", nulos_total)

    with col3:
        st.metric("Tratamientos únicos", len(df_filtrado['Tratamiento'].unique()))

    # Expanders para información detallada
    with st.expander("Valores nulos por columna"):
        nulos = df_filtrado.isnull().sum()
        st.dataframe(nulos, use_container_width=True)

    with st.expander("Tipos de datos"):
        tipos = df_filtrado.dtypes
        st.dataframe(tipos, use_container_width=True)

    with st.expander("Estadísticas descriptivas del Peso"):
        st.dataframe(df_filtrado['Peso_g'].describe(), use_container_width=True)

    # ===== SECCIÓN 2: BALANCEAMIENTO =====
    st.header("Balanceamiento del Experimento")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Observaciones por tratamiento")
        conteo_tratamientos = df_filtrado['Tratamiento'].value_counts()
        st.dataframe(conteo_tratamientos, use_container_width=True)

    with col2:
        st.subheader("Árboles únicos por tratamiento")
        arboles_por_tratamiento = df_filtrado.groupby('Tratamiento')['Arbol_ID'].nunique()
        st.dataframe(arboles_por_tratamiento, use_container_width=True)


    # ===== SECCIÓN 4: VISUALIZACIONES =====
    st.header("Visualizaciones")

    # Tabs para diferentes tipos de gráficas
    tab1, tab2, tab3, tab4 = st.tabs([
        "Violin Plot", "Histogramas", "Strip Plot", "Boxplot"
    ])


    custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FEA47F', '#F97F51', 
                    '#B33771', '#3B3B98', '#FD7272', '#9AECDB', '#BDC581', '#EAB543']

    with tab1:
        st.subheader("Violin Plot Interactivo")
        fig = px.violin(df_filtrado, x="Tratamiento", y="Peso_g", 
                    color="Tratamiento", 
                    color_discrete_sequence=custom_colors,  # Añadir custom colors
                    box=True, points="all",
                    title="Distribución de Peso por Tratamiento")
        fig.update_layout(showlegend=False)  # Opcional: ocultar leyenda si no es necesaria
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Histogramas por Tratamiento")
        
        # Crear subplots para histogramas con Plotly
        tratamientos_unicos = df_filtrado['Tratamiento'].unique()
        n_tratamientos = len(tratamientos_unicos)
        
        cols = 2
        rows = (n_tratamientos + 1) // cols
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f'Tratamiento: {tratamiento}' for tratamiento in tratamientos_unicos]
        )
        
        for i, tratamiento in enumerate(tratamientos_unicos):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            datos_tratamiento = df_filtrado[df_filtrado['Tratamiento'] == tratamiento]['Peso_g']
            
            # Usar el color correspondiente de la paleta personalizada
            color_idx = i % len(custom_colors)  # Para ciclar colores si hay más tratamientos que colores
            
            fig.add_trace(
                go.Histogram(
                    x=datos_tratamiento,
                    nbinsx=15,
                    name=f'{tratamiento}',
                    marker_color=custom_colors[color_idx],  # Aplicar color personalizado
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=400 * rows,
            title_text="Distribución de Pesos por Tratamiento",
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Peso (g)")
        fig.update_yaxes(title_text="Frecuencia")
        
        st.plotly_chart(fig, use_container_width=True)



        with tab3:
            st.subheader("Strip Plot por Tratamiento")
            
            fig = px.strip(
                df_filtrado,
                x="Tratamiento",
                y="Peso_g",
                title="Distribución de Pesos - Strip Plot",
                color="Tratamiento",
                color_discrete_sequence=custom_colors  # Usar paleta personalizada
            )
            
            fig.update_layout(
                xaxis_title="Tratamiento",
                yaxis_title="Peso (g)",
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Distribución del Peso por Tratamiento")
            
            fig = px.box(
                df_filtrado,
                x="Tratamiento",
                y="Peso_g",
                title="Distribución de Pesos - Diagrama de Cajas",
                color="Tratamiento",
                color_discrete_sequence=custom_colors  # Usar misma paleta personalizada
            )
            
            fig.update_traces(boxpoints='all', jitter=0.3, pointpos=-1.8)
            
            fig.update_layout(
                xaxis_title="Tratamiento",
                yaxis_title="Peso (g)",
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)









    # ===== SECCIÓN 5: ESTADÍSTICAS POR TRATAMIENTO =====
    st.header("Estadísticas Detalladas por Tratamiento")

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
        
        st.header("Modelo Jerárquico - Efectos Mixtos")

        st.markdown("""
        ### Modelo Ajustado
        El modelo estimado es: 
        $$Y_{ijk} = \\mu + \\tau_i + \\epsilon_{ij} + \\delta_{ijk}$$

        Donde:
        - $Y_{ijk}$: Peso del aguacate $k$ en el árbol $j$ del tratamiento $i$
        - $\\mu$: Media general
        - $\\tau_i$: Efecto fijo del tratamiento $i$
        - $\\epsilon_{ij}$: Efecto aleatorio del árbol $j$ dentro del tratamiento $i$
        - $\\delta_{ijk}$: Error aleatorio (variación entre aguacates del mismo árbol)
        """)
        ####
        with st.spinner("Ajustando modelo jerárquico..."):
            try:
                df_modelo = df.copy()
        
                df_modelo['Tratamiento'] = pd.Categorical(df_modelo['Tratamiento'], 
                                                        categories=['Control', 'Bio-A (Aminoacidos)', 
                                                                    'Bio-B (Ext. Algas)', 'Bio-C (Ac. Humicos)'])

                modelo_jerarquico = MixedLM.from_formula(
                    'Peso_g ~ Tratamiento', 
                    groups='Arbol_ID',
                    data=df_modelo
                )
                resultado_jerarquico = modelo_jerarquico.fit()
                
                st.subheader("Efectos Fijos - Tratamientos")
                
                coef_names = list(resultado_jerarquico.params.index)
                tratamiento_base = 'Control'
                
                tratamientos_list = []
                efectos_list = []
                pvalues_list = []
                
                tratamientos_list.append(tratamiento_base)
                efectos_list.append(resultado_jerarquico.params['Intercept'])
                pvalues_list.append(resultado_jerarquico.pvalues['Intercept'])
                
                for coef_name in coef_names:
                    if coef_name.startswith('Tratamiento[T.'):
                        trat_name = coef_name.replace('Tratamiento[T.', '').rstrip(']')
                        tratamientos_list.append(trat_name)
                        efectos_list.append(resultado_jerarquico.params['Intercept'] + resultado_jerarquico.params[coef_name])
                        pvalues_list.append(resultado_jerarquico.pvalues[coef_name])

                coef_df = pd.DataFrame({
                    'Tratamiento': tratamientos_list,
                    'Efecto (g)': efectos_list,
                    'Valor p': pvalues_list,
                    'Significativo': ['✓' if p < nivel_significancia else '✗' for p in pvalues_list]
                })
                
                st.dataframe(coef_df.style.format({
                    'Efecto (g)': '{:.2f}',
                    'Valor p': lambda x: format_p_value(x)
                }), 
                use_container_width=True)
                
                st.info("Interpretación: Control es el tratamiento de referencia. Los valores muestran el peso promedio esperado para cada tratamiento.")
                                
                var_arboles = resultado_jerarquico.cov_re.iloc[0,0]
                var_residual = resultado_jerarquico.scale
                var_total = var_arboles + var_residual
                                
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.subheader("Componentes de Varianza")
                    
                    # Mostrar métricas en vertical
                    st.metric(
                        "Varianza entre Árboles (σ²_ε)", 
                        f"{var_arboles:.2f} g²",
                        help="Cuánto varían los árboles entre sí"
                    )
                    
                    st.metric(
                        "Varianza Residual (σ²_δ)", 
                        f"{var_residual:.2f} g²",
                        help="Variación entre aguacates del mismo árbol"
                    )
                    
                    prop_var_arboles = var_arboles / var_total
                    st.metric(
                        "Correlación Intraclase (ICC)", 
                        f"{prop_var_arboles:.1%}",
                        help="Qué porcentaje de la variación total se debe a diferencias entre árboles"
                    )

                with col2:
                    st.subheader("Descomposición de la Varianza Total")
                    varianzas = [var_arboles, var_residual]
                    labels = [f'Entre Árboles:', 
                            f'Residual:']
                    colors = ['#FF6B6B', '#4ECDC4']
                    
                    fig = px.pie(
                        values=varianzas,
                        names=labels,
                        color_discrete_sequence=colors,
                        hole=0.3,  # Para hacer un donut chart (opcional)
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate="<b>%{label}</b><br>Varianza: %{value:.2f} g²<br>Porcentaje: %{percent}",
                    )
                    
                    fig.update_layout(
                        title={
                            'text': 'Composición de Varianza',
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 16, 'weight': 'bold'}
                        },
                        showlegend=False,
                        height=400,
                        margin=dict(t=50, b=20, l=20, r=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)


                
                st.subheader("Resumen Técnico del Modelo")
                
                summary_data = {
                    'Métrica': [
                        'Modelo',
                        'Método de estimación',
                        'Número de observaciones',
                        'Número de grupos (árboles)',
                        #'Log-Likelihood',
                        'AIC',
                        'BIC',
                        'Convergencia'
                    ],
                    'Valor': [
                        'Mixed Linear Model',
                        'REML',
                        f"{resultado_jerarquico.nobs:.0f}",
                        f"{df_modelo['Arbol_ID'].nunique()}",
                        #f"{resultado_jerarquico.llf:.2f}",
                        f"{resultado_jerarquico.aic:.2f}",
                        f"{resultado_jerarquico.bic:.2f}",
                        'Sí' if resultado_jerarquico.converged else 'No'
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                coef_completo = pd.DataFrame({
                    'Parámetro': resultado_jerarquico.params.index,
                    'Coeficiente': resultado_jerarquico.params.values,
                    'Error Estándar': resultado_jerarquico.bse.values,
                    'z': resultado_jerarquico.tvalues.values,
                    'P>|z|': resultado_jerarquico.pvalues.values,
                    'IC 95% Inferior': resultado_jerarquico.conf_int()[0].values,
                    'IC 95% Superior': resultado_jerarquico.conf_int()[1].values
                })
                
                st.dataframe(coef_completo.style.format({
                    'Coeficiente': '{:.4f}',
                    'Error Estándar': '{:.4f}',
                    'z': '{:.3f}',
                    'P>|z|': lambda x: format_p_value(x),
                    'IC 95% Inferior': '{:.4f}',
                    'IC 95% Superior': '{:.4f}'
                }), use_container_width=True, hide_index=True)
                        
            except Exception as e:
                st.error(f"Error en el modelo jerárquico: {str(e)}")

    # ===== SECCIÓN: ÁRBOLES CON VALORES EXTREMOS POR TRATAMIENTO =====
    st.header(" Árboles con Valores Extremos por Tratamiento")

    # Calcular promedios por árbol
    promedios_arboles = df_filtrado.groupby(['Tratamiento', 'Arbol_ID'])['Peso_g'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).round(2).reset_index()
    
    promedios_arboles.columns = ['Tratamiento', 'Arbol_ID', 'Promedio_Peso', 'Desviacion', 'Muestras', 'Minimo', 'Maximo']
    
    # Encontrar árboles con valores extremos por tratamiento
    resultados_extremos = []
    
    for tratamiento in df_filtrado['Tratamiento'].unique():
        datos_tratamiento = promedios_arboles[promedios_arboles['Tratamiento'] == tratamiento]
        
        if len(datos_tratamiento) > 0:
            # Árbol con mayor promedio
            arbol_max = datos_tratamiento.loc[datos_tratamiento['Promedio_Peso'].idxmax()]
            resultados_extremos.append({
                'Tratamiento': tratamiento,
                'Tipo': 'MAYOR Promedio',
                'Arbol_ID': arbol_max['Arbol_ID'],
                'Peso_Promedio (g)': arbol_max['Promedio_Peso'],
                'Desviacion (g)': arbol_max['Desviacion'],
                'Muestras': arbol_max['Muestras'],
                'Rango (g)': f"{arbol_max['Minimo']}-{arbol_max['Maximo']}"
            })
            
            # Árbol con menor promedio
            arbol_min = datos_tratamiento.loc[datos_tratamiento['Promedio_Peso'].idxmin()]
            resultados_extremos.append({
                'Tratamiento': tratamiento,
                'Tipo': 'MENOR Promedio',
                'Arbol_ID': arbol_min['Arbol_ID'],
                'Peso_Promedio (g)': arbol_min['Promedio_Peso'],
                'Desviacion (g)': arbol_min['Desviacion'],
                'Muestras': arbol_min['Muestras'],
                'Rango (g)': f"{arbol_min['Minimo']}-{arbol_min['Maximo']}"
            })

    # Mostrar resultados
    if resultados_extremos:
        df_extremos = pd.DataFrame(resultados_extremos)
        
        st.dataframe(
            df_extremos.style.format({
                'Peso_Promedio (g)': '{:.2f}',
                'Desviacion (g)': '{:.2f}'
            }).apply(
                lambda x: ['background-color: #E8F5E8' if x['Tipo'] == 'MAYOR Promedio' 
                        else 'background-color: #FFE8E8' for _ in x], 
                axis=1
            ),
            use_container_width=True
        )
        
        # Métricas rápidas
        col1, col2 = st.columns(2)
        
        with col1:
            mejor_arbol = df_extremos[df_extremos['Tipo'] == 'MAYOR Promedio'].loc[
                df_extremos['Peso_Promedio (g)'].idxmax()
            ]
            st.metric(
                " Mejor Árbol", 
                f"{mejor_arbol['Peso_Promedio (g)']}g",
                f"{mejor_arbol['Tratamiento']} - Árbol {mejor_arbol['Arbol_ID']}"
            )
        
        with col2:
            peor_arbol = df_extremos[df_extremos['Tipo'] == 'MENOR Promedio'].loc[
                df_extremos['Peso_Promedio (g)'].idxmin()
            ]
            st.metric(
                "Peor Árbol", 
                f"{peor_arbol['Peso_Promedio (g)']}g",
                f"{peor_arbol['Tratamiento']} - Árbol {peor_arbol['Arbol_ID']}"
            )
    else:
        st.warning("No hay datos suficientes para mostrar árboles extremos.")



    # ===== SECCIÓN 7: DATOS CRUDOS =====
    with st.expander("Ver Datos Crudos"):
        st.dataframe(df_filtrado, use_container_width=True)

        # Opción para descargar datos filtrados
        csv = df_filtrado.to_csv(index=False)
        st.download_button(
            label="Descargar datos filtrados como CSV",
            data=csv,
            file_name="datos_aguacate_filtrados.csv",
            mime="text/csv"
        )






else:
    # Información del Modelo Jerárquico - Reemplaza la bienvenida
    st.markdown(r"""
    ## Modelo Jerárquico de Efectos Mixtos para Datos Anidados

    ### Estructura Matemática del Modelo
    $$
    Y_{ijk} = \mu + \tau_i + \epsilon_{ij} + \delta_{ijk}
    $$

    **Componentes del modelo:**
    - $Y_{ijk}$: Observación $k$ en la unidad $j$ del tratamiento $i$
    - $\mu$: Media general de la población
    - $\tau_i$: Efecto fijo del tratamiento $i$
    - $\epsilon_{ij}$: Efecto aleatorio de la unidad experimental $j$ dentro del tratamiento $i$
    - $\delta_{ijk}$: Error de medición o variación residual dentro de la misma unidad experimental

    ### Aplicaciones en Diferentes Disciplinas

    #### 1. Ciencias de la Salud - Ensayos Clínicos Longitudinales
    Una de las aplicaciones de este modelo es en estudios médicos donde se realizan múltiples mediciones a lo largo del tiempo en los mismos pacientes. Permite separar el efecto real del tratamiento de la variabilidad natural entre pacientes y las fluctuaciones temporales dentro de cada individuo, proporcionando estimaciones más precisas de la eficacia terapéutica.

    #### 2. Educación - Evaluación de Programas Educativos
    En investigación educativa, el modelo jerárquico es fundamental para evaluar intervenciones pedagógicas cuando los estudiantes son evaluados múltiples veces bajo diferentes condiciones. Distingue entre la efectividad real del programa educativo y las diferencias inherentes entre estudiantes, evitando conclusiones erróneas sobre la efectividad de las metodologías de enseñanza.

    #### 3. Control de Calidad Industrial
    En entornos de manufactura, este modelo permite analizar procesos donde se toman múltiples mediciones por lote de producción. Identifica si las variaciones en la calidad se deben a diferencias entre lotes o a fluctuaciones dentro del mismo lote, facilitando la optimización de parámetros de proceso y el mejoramiento continuo.
    
    
    """)
    