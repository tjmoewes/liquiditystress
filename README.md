# Random Forest — Tensión de Liquidez

Proyecto académico de Machine Learning aplicado a administración de capital de trabajo.

## Archivos

- `app.py` — interfaz y ejecución en Streamlit.
- `random_forest_tension_liquidez.py` — motor del modelo Random Forest.
- `Base_Didactica_Random_Forest_Capital_Trabajo.xlsx` — base de datos.
- `requirements.txt` — dependencias.
- `README.md` — instrucciones.

## Configuración del modelo

- Modelo: Random Forest Classifier
- Target: `Tension_Liquidez_bin`
- Train/Test: 70% / 30%
- Stratify: Sí
- `random_state`: 42
- Predictores: variables financieras permitidas
- Exclusiones: `ID_Observacion`, `Fecha`, `Prob_Tension_Liquidez` y el target
- Variables categóricas: One-Hot Encoding
- Faltantes numéricos: mediana
- Faltantes categóricos: moda
- `n_estimators`: 100
- `max_depth`: 8
- `class_weight`: `balanced`
- Validación: Train/Test
- Cross-Validation: No
- Threshold: 0.50
- Métrica principal: Recall de clase 1
- Métricas adicionales: Accuracy, Precision, F1 y ROC-AUC
- Feature importance: Sí
- Matriz de confusión: Sí
- Curva ROC: Sí

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit

1. Crear un repositorio nuevo en GitHub.
2. Subir todos los archivos de este proyecto.
3. En Streamlit, crear una nueva aplicación conectada al repositorio.
4. Seleccionar `app.py` como archivo principal.
5. Desplegar.

No es necesario subir el archivo `.pkl` generado: la aplicación entrena el modelo al ejecutarse.
