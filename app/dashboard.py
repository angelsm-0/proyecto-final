import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import io

st.set_page_config(page_title="Fundación Canguro - Predictor de Malnutrición", layout="wide")

# CSS Premium Styling
st.markdown("""
    <style>
    /* Asegurar fondo claro y texto oscuro de alto contraste en el área principal */
    .main {
        background-color: #f4f7f6 !important;
    }
    .main p, .main span, .main label, .main li, .main div {
        color: #333333;
    }
    
    /* Forzar títulos legibles en color azul marino */
    h1, h2, h3, h4, h5, h6 {
        color: #1e3d59 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Asegurar que las tarjetas st.metric tengan fondo blanco y texto oscuro de alta legibilidad */
    .stMetric {
        background-color: #ffffff !important;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stMetricValue"] {
        color: #1e3d59 !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
        font-weight: 500 !important;
    }
    
    /* Botones de acción */
    .stButton>button {
        background-color: #ff6e40 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #ff8a65 !important;
        color: white !important;
    }
    
    /* Barra lateral (Sidebar) en azul oscuro con texto e íconos en blanco para máximo contraste */
    [data-testid="stSidebar"] {
        background-color: #1e3d59 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] li, [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Cajas y Alertas del Método Madre Canguro con textos oscuros forzados */
    .kmm-card {
        background-color: #e3f2fd !important;
        color: #0d47a1 !important;
        border-left: 6px solid #2196f3 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        margin-top: 15px !important;
    }
    .kmm-card strong {
        color: #0d47a1 !important;
    }
    .kmm-success {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        border-left: 6px solid #4caf50 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        margin-top: 15px !important;
    }
    .kmm-success strong {
        color: #1b5e20 !important;
    }
    .kmm-warning {
        background-color: #fff3e0 !important;
        color: #e65100 !important;
        border-left: 6px solid #ff9800 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        margin-top: 15px !important;
    }
    .kmm-warning strong {
        color: #e65100 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar Datos y Benchmark
@st.cache_data
def load_data():
    return pd.read_parquet("../modelos/KMC_cleaned.parquet")

@st.cache_data
def load_benchmark():
    return pd.read_csv("phase_comparison_results_v2.csv")

df = load_data()
comp_df = load_benchmark()

# Sidebar Navigation (Vistas del Sistema)
# st.sidebar.image("", width=180)
st.sidebar.markdown("## 🧭 Navegación")
vista = st.sidebar.radio("Seleccione una Vista:", [
    "📊 Analítica Poblacional", 
    "🔮 Evaluación de Pacientes Individuales", 
    "📂 Carga y Predicción en Lote (Cohortes)"
])



# ==========================================
# VISTA 1: ANALÍTICA POBLACIONAL (DASHBOARD)
# ==========================================
if vista == "📊 Analítica Poblacional":
    st.title("📊 Analítica Poblacional e Histórica Canguro")
    st.markdown("### Dashboard Clínico de Monitoreo de Malnutrición")
    st.markdown("---")
    
    # Sidebar de Filtros Poblacionales
    st.sidebar.header("🔍 Filtros de Cohorte Histórica")
    period = st.sidebar.multiselect("Periodo de Nacimiento (Quinquenios)", 
                                    options=df['Grupo_Period'].unique() if 'Grupo_Period' in df.columns else df['Grupo_Periodo'].unique(), 
                                    default=df['Grupo_Periodo'].unique())
    mal_cat = st.sidebar.multiselect("Categorías Clínicas de Malnutrición", 
                                     options=df['Categoria_Malnutricion'].unique(), 
                                     default=df['Categoria_Malnutricion'].unique())
    
    filtered_df = df[(df['Grupo_Periodo'].isin(period)) & (df['Categoria_Malnutricion'].isin(mal_cat))]
    
    # Métricas Principales de Población
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 Total de Infantes", f"{len(filtered_df):,}")
    col2.metric("📊 Prevalencia Malnutrición", f"{(filtered_df['Cualquier_Malnutricion'].mean() * 100):.1f}%")
    col3.metric("⚖️ Promedio Peso al Nacer", f"{filtered_df['ERN_Peso'].mean():.1f} g")
    col4.metric("👶 Días Posición Canguro (Med.)", f"{int(filtered_df['AC_DiasPosCanguro'].median())} días")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Distribución y Tendencias", "🧬 Benchmarking de Modelos"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Distribución de Categorías de Malnutrición")
            fig_pie = px.pie(filtered_df, names='Categoria_Malnutricion', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333333')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_right:
            st.subheader("Tendencia Temporal de Prevalencia por Quinquenios")
            trend = df.groupby('Grupo_Periodo')['Cualquier_Malnutricion'].mean().reset_index()
            fig_trend = px.line(trend, x='Grupo_Periodo', y='Cualquier_Malnutricion', markers=True,
                                labels={'Cualquier_Malnutricion': 'Prevalencia (%)', 'Grupo_Periodo': 'Quinquenio'})
            fig_trend.update_layout(
                yaxis_tickformat='.0%',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333333')
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
    with tab2:
        st.subheader("Poder Predictivo Acumulado por Fases Temporales")
        fig_comp = px.line(comp_df, x='Phase', y='AUC', markers=True, text='AUC',
                           title="Mejora en el Área Bajo la Curva (AUROC) al acumular datos longitudinales",
                           labels={'AUC': 'AUROC (Capacidad Predictiva)', 'Phase': 'Fase de Consulta'})
        fig_comp.update_traces(textposition="bottom right", texttemplate='%{text:.3f}', line_color='#1e3d59', line_width=3)
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333')
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("""
            > **Interpretación Predictiva de Maestría:** El incremento sustancial del AUROC de **0.739 a 0.849** entre la fase del **Nacimiento** y la consulta de los **3 Meses** marca el *punto de inflexión*. 
            Esto demuestra matemáticamente que la velocidad de crecimiento post-natal y la adaptación extrauterina son altamente informativas para predecir el desenlace del primer año.
        """)

# ====================================================
# VISTA 2: EVALUACIÓN DE PACIENTES INDIVIDUALES
# ====================================================
elif vista == "🔮 Evaluación de Pacientes Individuales":
    st.title("🔮 Consulta y Evaluación Clínica Individual")
    # Footer removed per user request
    st.markdown("---")
    
    st.markdown("Esta consola interactiva permite al pediatra ingresar los datos capturados de un infante en la consulta actual y evaluar la probabilidad de riesgo de malnutrición a los 12 meses de edad corregida.")
    
    st.markdown("---")
    
    # Selector de Fase del Paciente
    fase_seleccionada = st.selectbox(
        "Seleccione la Fase de Consulta del Paciente:",
        ["Nacimiento", "40 Semanas", "3 Meses"]
    )
    
    nombre_archivo_base = fase_seleccionada.replace(" ", "_").lower()
    ruta_modelo = f"../modelos/modelo_{nombre_archivo_base}.joblib"
    ruta_json = f"../modelos/features_{nombre_archivo_base}.json"
    
    if os.path.exists(ruta_modelo) and os.path.exists(ruta_json):
        # Cargar modelo y metadatos
        modelo = joblib.load(ruta_modelo)
        with open(ruta_json, "r") as f:
            meta = json.load(f)
            
        features = meta["features"]
        medians = meta["medians"]
        
        # Valores por defecto basados en preset o mediana de entrenamiento
        def_peso_nacer = st.session_state.get("peso_nacer", 1500)
        def_talla_nacer = st.session_state.get("talla_nacer", 40)
        def_sexo = st.session_state.get("sexo", 2)
        def_talla_madre = st.session_state.get("talla_madre", 158)
        def_talla_padre = st.session_state.get("talla_padre", 168)
        def_embarazo_multiple = st.session_state.get("embarazo_multiple", 1)
        def_peso_40w = st.session_state.get("peso_40w", 2500)
        def_talla_40w = st.session_state.get("talla_40w", 46)
        def_peso_3m = st.session_state.get("peso_3m", 4200)
        def_talla_3m = st.session_state.get("talla_3m", 52)
        def_dias_canguro = st.session_state.get("dias_canguro", 15)
        
        # Estructura del Formulario según la fase
        col_input, col_output = st.columns([1.2, 1])
        
        with col_input:
            st.markdown(f"### 📋 Variables Clínicas de la Fase: **{fase_seleccionada}**")
            
            # Variables Neonatales / Nacimiento
            peso_nacer = st.slider("Peso al Nacer (gramos)", min_value=200, max_value=5000, value=def_peso_nacer, step=50, key="slide_peso_nacer")
            talla_nacer = st.slider("Talla al Nacer (cm)", min_value=30, max_value=55, value=def_talla_nacer, step=1, key="slide_talla_nacer")
            sexo = st.selectbox("Sexo del Neonato", options=[("Masculino", 1), ("Femenino", 2)], index=def_sexo - 1, format_func=lambda x: x[0], key="slide_sexo")[1]
            
            # Contexto Genético y Gestacional (Variables muy importantes bajo SHAP)
            with st.expander("🧬 Contexto Genético y Gestacional"):
                st.markdown("Estos factores hereditarios y gestacionales influyen fuertemente en el canal de crecimiento del lactante:")
                talla_madre = st.slider("Talla de la Madre (cm)", min_value=135, max_value=190, value=def_talla_madre, step=1, key="slide_talla_madre")
                talla_padre = st.slider("Talla del Padre (cm)", min_value=145, max_value=205, value=def_talla_padre, step=1, key="slide_talla_padre")
                embarazo_multiple = st.selectbox("Tipo de Embarazo", options=[("Sencillo", 1), ("Doble (Gemelar)", 2), ("Triple o más", 3)], index=def_embarazo_multiple - 1, format_func=lambda x: x[0], key="slide_embarazo")[1]
            
            # Días Kangaroo (KMM) - Intervención del programa
            dias_canguro = st.slider("Días en Posición Canguro Piel a Piel", min_value=0, max_value=200, value=def_dias_canguro, step=1, key="slide_dias_canguro")
            
            # Variables de 40 Semanas si aplica
            peso_40w, talla_40w = 2500, 46
            if fase_seleccionada == "40 Semanas":
                st.markdown("---")
                st.markdown("#### 👶 Controles a las 40 Semanas")
                peso_40w = st.slider("Peso a las 40 Semanas (gramos)", min_value=1000, max_value=4500, value=def_peso_40w, step=50, key="slide_peso_40w")
                talla_40w = st.slider("Talla a las 40 Semanas (cm)", min_value=35, max_value=60, value=def_talla_40w, step=1, key="slide_talla_40w")
                
            # Variables de 3 Meses si aplica
            peso_3m, talla_3m = 4200, 52
            if fase_seleccionada == "3 Meses":
                st.markdown("---")
                st.markdown("#### 👶 Controles a los 3 Meses")
                peso_3m = st.slider("Peso a los 3 Meses (gramos)", min_value=1500, max_value=7000, value=def_peso_3m, step=50, key="slide_peso_3m")
                talla_3m = st.slider("Talla a los 3 Meses (cm)", min_value=40, max_value=70, value=def_talla_3m, step=1, key="slide_talla_3m")
        
        with col_output:
            st.markdown("### 📊 Evaluación Predictiva de Riesgo")
            
            # Crear Fila de Predicción poblada con medianas de entrenamiento
            row_dict = medians.copy()
            
            # Sobreescribir con inputs del pediatra
            row_dict['ERN_Peso'] = peso_nacer
            row_dict['ERN_Talla'] = talla_nacer
            #row_dict['V219'] = talla_nacer
            row_dict['ERN_Sexo'] = sexo
            row_dict['CP_TallaMadre'] = talla_madre
            row_dict['CP_TallaPadre'] = talla_padre
            row_dict['Iden_embarazoMultiple'] = embarazo_multiple
            row_dict['AC_DiasPosCanguro'] = dias_canguro
            row_dict['Grupo_Periodo_Codificado'] = 5  # Por defecto al quinquenio más moderno para nuevos pacientes (2020-2022)
            
            if fase_seleccionada == "40 Semanas":
                row_dict['V219'] = talla_40w
                row_dict['V218'] = peso_40w
                
                # Calcular Z-Scores aproximados si aplica
                
            if fase_seleccionada == "3 Meses":
                row_dict['V219'] = talla_40w
                row_dict['V261'] = peso_3m
                row_dict['V262'] = talla_3m
                # Calcular velocidades aproximadas si aplica
                
            # Alinear columnas con el modelo
            X_pred = pd.DataFrame([row_dict])[features]
            
            # Inferencia
            probabilidad = modelo.predict_proba(X_pred)[0, 1]
            
            # Configurar alertas visuales
            if probabilidad < 0.35:
                color_alerta = "green"
                categoria_riesgo = "🟢 BAJO RIESGO DE MALNUTRICIÓN"
            elif probabilidad < 0.65:
                color_alerta = "orange"
                categoria_riesgo = "🟡 RIESGO MODERADO - REQUIERE MONITOREO"
            else:
                color_alerta = "red"
                categoria_riesgo = "🔴 ALTO RIESGO DE MALNUTRICIÓN - ALERTA TEMPRANA"
                
            st.markdown(f"""
                <div style="background-color: white; border-radius: 12px; padding: 25px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-top: 8px solid {color_alerta};">
                    <h2 style="color: #1e3d59; font-weight: bold; margin: 0;">Probabilidad Calculada</h2>
                    <h1 style="color: {color_alerta}; font-size: 4rem; margin: 10px 0; font-weight: 800;">{probabilidad*100:.1f}%</h1>
                    <h3 style="color: #333333; font-weight: bold; margin: 0;">{categoria_riesgo}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # 🩺 Kangaroo Mother Method Adherence analysis
            st.markdown("#### 👶 Impacto del Método Madre Canguro (MMC)")
            if dias_canguro >= 20:
                st.markdown(f"""
                    <div class="kmm-success">
                        <strong>✅ Adherencia al Método Canguro Excelente:</strong> El bebé cuenta con {dias_canguro} días de contacto piel a piel. Esto estimula la succión, estabiliza el ritmo cardiorrespiratorio y acelera la ganancia de peso compensatoria, reduciendo considerablemente la vulnerabilidad del infante.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="kmm-warning">
                        <strong>⚠️ Alerta de Adherencia Canguro:</strong> El bebé registra solo {dias_canguro} días de posición piel a piel. Aumentar la adherencia (meta sugerida de >20 días) incrementará notablemente el desarrollo del infante y puede mitigar significativamente el riesgo predictivo.
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### 📋 Recomendaciones y Plan de Acción Clínico:")
            if probabilidad >= 0.65:
                st.error("🚨 **Intervención Prioritaria:** Agendar control extraordinario de crecimiento en 7 días, revisar la adherencia al Método Canguro Piel a Piel, auditar técnica de lactancia materna e iniciar suplementación calórica supervisada.")
            elif probabilidad >= 0.35:
                st.warning("⚠️ **Vigilancia Estrecha:** Programar control preventivo a las 2 semanas. Realizar monitoreo intensivo de ganancia de peso diaria (meta > 15g/kg/día).")
            else:
                st.success("✅ **Esquema Canguro Normal:** Continuar con los controles de rutina mensuales programados de forma regular.")
            
            # 📊 Explicabilidad Clínica (Feature Importance)
            st.markdown("---")
            st.markdown("### 🧠 Explicabilidad Clínica y Transparencia")
            st.markdown(f"A continuación se presentan las variables de mayor impacto relativo para el modelo de la fase **{fase_seleccionada}** (basado en análisis SHAP y Gini):")
            
            # Features and importances according to phase
            if fase_seleccionada == "Nacimiento":
                imp_features = ["Talla de la madre", "Peso al Nacer", "Sexo al nacer", "Talla al nacer"]
                imp_values = [44, 35, 13, 8]
            elif fase_seleccionada == "40 Semanas":
                imp_features = ["Talla 40 Semanas", "Talla Madre", "Talla Padre", "Peso al nacer"]
                imp_values = [39, 27, 19, 15]
            else: # 3 Meses
                imp_features = ["Peso a los 3 meses", "Talla a los 3 meses", "Sexo al nacer", "Talla a las 40 semanas"]
                imp_values = [31, 27, 26, 16]
                
            fig_shap = go.Figure(go.Bar(
                x=imp_values,
                y=imp_features,
                orientation='h',
                marker_color='#1e3d59',
                text=[f"{v}%" for v in imp_values],
                textposition='auto',
            ))
            fig_shap.update_layout(
                margin=dict(l=20, r=20, t=10, b=10),
                height=220,
                xaxis_title="Importancia Relativa (%)",
                yaxis=dict(autorange="reversed"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333333')
            )
            st.plotly_chart(fig_shap, use_container_width=True)
                
    else:
        st.error(f"El modelo binario serializado para la fase '{fase_seleccionada}' no está disponible. Corre primero el Jupyter Notebook.")

# ====================================================
# VISTA 3: CARGA Y PREDICCIÓN EN LOTE (BATCH PREDICTION)
# ====================================================
elif vista == "📂 Carga y Predicción en Lote (Cohortes)":
    st.title("📂 Evaluación Masiva de Cohortes")
    st.markdown("### Predicción del Riesgo de Malnutrición en Lote para Nuevos Pacientes")
    st.markdown("---")
    
    st.markdown("""
        Esta vista está diseñada para permitir al personal administrativo y clínico de la Fundación Canguro cargar planillas con datos de cohortes de nuevos infantes, 
        correr predicciones de riesgo de manera masiva y descargar un informe enriquecido en formato CSV.
    """)
    
    # 1. Selector de Modelo para el lote
    fase_modelo = st.selectbox("Seleccione el Modelo Clínico para evaluar la cohorte:", ["Nacimiento", "40 Semanas", "3 Meses"])
    
    nombre_archivo_base = fase_modelo.replace(" ", "_").lower()
    ruta_modelo = f"../modelos/modelo_{nombre_archivo_base}.joblib"
    ruta_json = f"../modelos/features_{nombre_archivo_base}.json"
    
    if os.path.exists(ruta_modelo) and os.path.exists(ruta_json):
        modelo = joblib.load(ruta_modelo)
        with open(ruta_json, "r") as f:
            meta = json.load(f)
        features = meta["features"]
        medians = meta["medians"]
        
        # 2. Descargar Plantilla de Ejemplo
        st.markdown("### 1. Descargar Plantilla Oficial")
        st.markdown("Asegúrese de que su archivo tenga exactamente las columnas listadas en la plantilla. El sistema rellenará de forma automática cualquier variable ausente utilizando las medianas exactas del entrenamiento clínico.")
        
        # Generar DataFrame muestra
        sample_data = {
            "ID_Paciente": [1001, 1002, 1003],
            "ERN_Peso": [1200, 1600, 950],
            "ERN_Talla": [38, 42, 34],
            "ERN_Sexo": [1, 2, 1],
            "CP_TallaMadre": [155, 160, 150]
        }
        
        if fase_modelo == "40 Semanas":
            sample_data["V218"] = [2500, 2900, 2200]
            sample_data["V219"] = [46, 48, 44]
        elif fase_modelo == "3 Meses":
            sample_data["V261"] = [3100, 4800, 2900]
            sample_data["V262"] = [48, 54, 46]
            
        df_sample = pd.DataFrame(sample_data)
        csv_sample = df_sample.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=f"📥 Descargar Plantilla CSV de Ejemplo ({fase_modelo})",
            data=csv_sample,
            file_name=f"plantilla_cohorte_{nombre_archivo_base}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # 3. Subidor de Archivo
        st.markdown("### 2. Cargar Planilla de Nuevos Pacientes")
        uploaded_file = st.file_uploader("Suba su archivo en formato CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file)
                st.success("Planilla de pacientes cargada correctamente.")
                st.write(f"Total de registros a procesar: **{len(df_new)}**")
                
                # Botón para ejecutar predicción masiva
                if st.button("🔮 Correr Inferencia Predictiva"):
                    with st.spinner("Procesando cohortes y ejecutando modelos..."):
                        # Construir la lista de filas con los valores del usuario y las medianas de entrenamiento
                        rows_to_predict = []
                        for idx, row in df_new.iterrows():
                            row_dict = medians.copy()
                            for col in df_new.columns:
                                if col in row_dict:
                                    row_dict[col] = row[col]
                            row_dict['Grupo_Periodo_Codificado'] = 5  # Por defecto al quinquenio más moderno para nuevos pacientes
                            rows_to_predict.append(row_dict)
                        
                        # Convertir a DataFrame todo el lote y alinear columnas en un solo paso
                        X_pred_all = pd.DataFrame(rows_to_predict)[features]
                        
                        # Imputación robusta de valores faltantes (NaN) usando las medianas del entrenamiento clínico
                        for col in X_pred_all.columns:
                            if X_pred_all[col].isna().any():
                                X_pred_all[col] = X_pred_all[col].fillna(medians.get(col, 0))
                        
                        # Inferencia masiva optimizada
                        resultados_lote = modelo.predict_proba(X_pred_all)[:, 1]
                        df_new['Probabilidad_Riesgo'] = resultados_lote
                        df_new['Porcentaje_Riesgo'] = df_new['Probabilidad_Riesgo'].apply(lambda x: f"{x*100:.1f}%")
                        
                        # Asignar categorías de riesgo
                        def categorizar(p):
                            if p < 0.35: return "🟢 Bajo Riesgo"
                            elif p < 0.65: return "🟡 Riesgo Moderado"
                            return "🔴 Alto Riesgo"
                            
                        def recomendar(p):
                            if p < 0.35: return "Esquema Canguro Normal"
                            elif p < 0.65: return "Control Preventivo (2 sem)"
                            return "Intervención Prioritaria (7 días)"
                            
                        df_new['Categoria_Riesgo'] = df_new['Probabilidad_Riesgo'].apply(categorizar)
                        df_new['Accion_Recomendada'] = df_new['Probabilidad_Riesgo'].apply(recomendar)
                        
                        st.markdown("### 3. Resultados de Inferencia Masiva")
                        
                        # Métricas Resumen del Lote
                        c_alto = sum(df_new['Probabilidad_Riesgo'] >= 0.65)
                        c_mod = sum((df_new['Probabilidad_Riesgo'] >= 0.35) & (df_new['Probabilidad_Riesgo'] < 0.65))
                        c_bajo = sum(df_new['Probabilidad_Riesgo'] < 0.35)
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔴 Casos de Alto Riesgo", f"{c_alto}")
                        c2.metric("🟡 Casos de Riesgo Moderado", f"{c_mod}")
                        c3.metric("🟢 Casos de Bajo Riesgo", f"{c_bajo}")
                        
                        # Mostrar tabla de resultados interactiva
                        show_cols = ['ID_Paciente', 'ERN_Peso', 'Porcentaje_Riesgo', 'Categoria_Riesgo', 'Accion_Recomendada']
                        available_show_cols = [c for c in show_cols if c in df_new.columns]
                        if 'ID_Paciente' not in df_new.columns:
                            # Add patient index as patient ID
                            df_new['ID_Paciente'] = df_new.index + 1000
                            available_show_cols.insert(0, 'ID_Paciente')
                            
                        st.dataframe(df_new[available_show_cols])
                        
                        # Generar descarga de CSV resultante
                        csv_output = df_new.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Informe Completo de Predicciones",
                            data=csv_output,
                            file_name=f"informe_riesgo_cohorte_{nombre_archivo_base}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error procesando el archivo CSV: {e}. Verifique que las columnas coincidan con la plantilla.")
    else:
        st.error(f"El modelo binario serializado para la fase '{fase_modelo}' no está disponible. Corre primero el Jupyter Notebook.")


