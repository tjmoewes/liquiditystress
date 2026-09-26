import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from random_forest_tension_liquidez import (
    TARGET,
    MODEL_CONFIG,
    load_data,
    train_random_forest,
    plot_confusion_matrix,
    plot_roc,
    plot_importance,
    importance_to_excel,
)

st.set_page_config(
    page_title="Random Forest - Tensión de Liquidez",
    page_icon="📊",
    layout="wide",
)

st.title("Random Forest para Tensión de Liquidez")
st.caption(
    "Modelo de clasificación aplicado a administración de capital de trabajo"
)

ARCHIVO = "Base_Didactica_Random_Forest_Capital_Trabajo.xlsx"

with st.sidebar:
    st.header("Configuración del modelo")
    st.write("**Modelo:** Random Forest Classifier")
    st.write(f"**Target:** `{TARGET}`")
    st.write("**Train / Test:** 70% / 30%")
    st.write("**Stratify:** Sí")
    st.write(f"**random_state:** {MODEL_CONFIG['random_state']}")
    st.write(f"**n_estimators:** {MODEL_CONFIG['n_estimators']}")
    st.write(f"**max_depth:** {MODEL_CONFIG['max_depth']}")
    st.write("**class_weight:** balanced")
    st.write(f"**Threshold:** {MODEL_CONFIG['threshold']:.2f}")
    st.write("**Métrica principal:** Recall clase 1")

@st.cache_data
def cargar_base():
    return load_data(ARCHIVO)

try:
    df = cargar_base()
except Exception as e:
    st.error(
        "No fue posible cargar la base de datos. "
        f"Verifica que `{ARCHIVO}` esté en el mismo repositorio que `app.py`."
    )
    st.exception(e)
    st.stop()

if TARGET not in df.columns:
    st.error(f"No se encontró la columna target `{TARGET}`.")
    st.stop()

# Resumen de la base
st.subheader("1. Base de datos")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observaciones", f"{len(df):,}")
c2.metric("Variables", f"{df.shape[1]:,}")
c3.metric("Train", f"{int(len(df) * 0.70):,}")
c4.metric("Test", f"{len(df) - int(len(df) * 0.70):,}")

with st.expander("Ver primeras observaciones"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Distribución del target"):
    st.dataframe(
        df[TARGET].value_counts()
        .rename_axis(TARGET)
        .reset_index(name="Observaciones"),
        use_container_width=True,
    )

# Entrenamiento
st.subheader("2. Entrenamiento y evaluación")

with st.spinner("Entrenando Random Forest..."):
    resultados = train_random_forest(df)

metrics = resultados["metrics"]

st.success("Modelo entrenado correctamente.")

# Métricas
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Recall", f"{metrics['Recall']:.3f}")
m2.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
m3.metric("Precision", f"{metrics['Precision']:.3f}")
m4.metric("F1", f"{metrics['F1']:.3f}")
m5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

st.info(
    "El Recall de la clase 1 es la métrica principal porque el objetivo es "
    "identificar observaciones con tensión de liquidez."
)

# Gráficas
st.subheader("3. Diagnóstico del modelo")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        plot_confusion_matrix(resultados["confusion_matrix"]),
        use_container_width=True,
    )

with col2:
    st.pyplot(
        plot_roc(resultados["y_test"], resultados["y_prob"]),
        use_container_width=True,
    )

with st.expander("Classification Report"):
    st.text(resultados["classification_report"])

# Importancia
st.subheader("4. Importancia de variables")

st.pyplot(
    plot_importance(resultados["importance"], top_n=15),
    use_container_width=True,
)

st.dataframe(
    resultados["importance"].head(15),
    use_container_width=True,
)

# Descargas
st.subheader("5. Archivos generados")

excel_buffer = importance_to_excel(resultados["importance"])

st.download_button(
    label="Descargar importancia de variables (Excel)",
    data=excel_buffer,
    file_name="importancia_variables_random_forest.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "La aplicación reentrena el modelo al cargar la página utilizando la "
    "configuración autorizada para el ejercicio."
)
