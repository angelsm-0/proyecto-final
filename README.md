# 📊 Malnutrición Infantil – Predictor de Riesgo

## Descripción general
Este proyecto implementa un modelo de aprendizaje automático que predice el riesgo de malnutrición en lactantes a los 12 meses de edad corregida, a partir de variables clínicas y demográficas recopiladas en la primera visita pediátrica.  La solución se despliega como una aplicación **Streamlit** que permite tanto la carga masiva como la evaluación individual de pacientes.

## Instrucciones básicas de uso
1. **Clonar el repositorio**.
2. Crear un entorno virtual e instalar dependencias:
   ```bash
   pip install -r requirements_dashboard.txt
   # Para entrenamiento (opcional)
   pip install -r requirements_training.txt
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
   web: streamlit run dashboard.py --server.port $PORT
   ```

## Ejemplos de uso
### Evaluación de cohorte (CSV)
1. Subir un archivo CSV con la estructura indicada en `data/template_carga.csv`.
2. La aplicación genera `informe_riesgo_cohorte_*.csv` con una columna adicional `riesgo_predicho` (valor entre 0‑1).

### Evaluación individual
```python
# En la barra lateral seleccionar "🔮 Evaluación de Pacientes Individuales"
# Ingresar los valores solicitados (edad, peso, altura, etc.)
# Pulsar "Predecir riesgo"
# La app muestra la probabilidad y un mensaje de recomendación.
```

