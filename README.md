# 📊 Malnutrición Infantil – Predictor de Riesgo

## Descripción general
Este proyecto implementa un modelo de aprendizaje automático que predice el riesgo de malnutrición en lactantes a los 12 meses de edad corregida, a partir de variables clínicas y demográficas recopiladas en la primera visita pediátrica.  La solución se despliega como una aplicación **Streamlit** que permite tanto la carga masiva como la evaluación individual de pacientes.

## Instrucciones básicas de uso
1. **Clonar el repositorio**.
2. Crear un entorno virtual e instalar dependencias:
   ```bash
   pip install -r requirements_dashboard.txt
   # Para entrenamiento (opcional)
   pip install -r requirements_entrenamieto.txt
   ```
3. Ejecutar la aplicación:
   ```bash
   streamlit run dashboard.py
   ```
   La aplicación se abrirá en el navegador en `http://localhost:8501`.

## Enlaces relevantes
- **Aplicación desplegada (AWS)**: http://44.209.187.21:8501/
- **Repositorio del código**: https://github.com/angelsm-0/proyecto-final
- **Datasets de entrenamiento**: `data/` contiene los archivos Parquet y CSV utilizados en los notebooks.  Los archivos originales pueden solicitarse a la fundación bajo acuerdo de uso.
- **Documentación adicional**: `docs/` (si existiese) o el manual de usuario generado en `Manual_Usuario_App_Malnutricion_Canguro.docx`.

## Dependencias y entorno de ejecución
- **Python ≥ 3.9**
- **Archivo de dependencias**:
  - `requirements_dashboard.txt` – paquetes para la UI y visualizaciones.
  - `requirements_training.txt` – paquetes para entrenamiento y exportación de modelos.

## Pasos para reproducir el despliegue
1. **Instalar** los requisitos como se indica arriba.
2. **Descargar** los modelos entrenados (`modelo_*.joblib`) que ya están incluidos en la carpeta raíz.
3. **Ejecutar** `streamlit run dashboard.py`.
4. Para **desplegar en AWS** (EC2) basta con empaquetar la carpeta y ejecutar el mismo comando dentro del contenedor; la configuración de `Procfile` típicamente contiene:
   ```text
   web: streamlit run dashboard.py
   
   ```
##  Ejemplos de uso (inputs/outputs esperados)
### upload
#### ID_Paciente,Fecha_Nacimiento,Sexo,CP_TallaMadre,CP_PesoMadre,CP_TallaPadre,CP_PesoPadre,ERN_Peso,ERN_Talla,RCIUtalla,V219,V261,V262,ERN_Sexo
- 001,2022-02-15,F,158,60,165,70,3.4,48,0.85,84,78,73,1
- 002,2022-01-10,M,162,62,168,75,3.2,46,0.80,71,88,78,0
- 003,2022-03-05,F,155,58,160,68,3.5,49,0.90,90,85,80,1

### output
#### ID_Paciente,Probabilidad_Riesgo,Clasificacion,Riesgo_Alto_Threshold,Modelo_Utilizado
- 001,0.78,Riesgo Alto,0.20,modelo_3_meses
- 002,0.12,Bajo Riesgo,0.20,modelo_3_meses
- 003,0.54,Riesgo Alto,0.20,modelo_3_meses

### Evaluación de cohorte (CSV)
1. Subir un archivo CSV con la estructura indicada en `data/template_carga.csv`.
2. La aplicación genera `informe_riesgo_cohorte_*.csv` con una columna adicional `riesgo_predicho` (valor entre 0‑1).

### Evaluación individual

1. En la barra lateral seleccionar "🔮 Evaluación de Pacientes Individuales"
2. Ingresar los valores solicitados (edad, peso, altura, etc.)
3. Pulsar "Predecir riesgo"
4. La app muestra la probabilidad y un mensaje de recomendación.

##  Consideraciones sobre el modelo

Requisitos mínimos para ejecutar los modelos
Los tres modelos (3 meses, nacimiento y 40 semanas) están entrenados con algoritmos de boosting basados en árboles (XGBoost y LightGBM) y cada archivo .joblib ocupa menos de 70 KB.

### Hardware
1. CPU: al menos 2 vCPU (una instancia t3.small de AWS es suficiente).
2. Memoria RAM: 400 – 500 MB por modelo; con los tres cargados simultáneamente basta con 2 GiB de RAM.
3. Almacenamiento: 1 GB de disco (EBS gp3) cubre con holgura los modelos y sus metadatos.
4. GPU: no requerida; la inferencia en CPU es < 0.05 s por mil registros.


