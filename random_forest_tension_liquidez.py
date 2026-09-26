# ============================================================
# MODELO 1 - RANDOM FOREST PARA TENSION DE LIQUIDEZ
# Configuración autorizada
# ============================================================

from pathlib import Path
import io
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Tension_Liquidez_bin"

EXCLUIR = [
    "ID_Observacion",
    "Fecha",
    "Prob_Tension_Liquidez",
    TARGET,
]

MODEL_CONFIG = {
    "test_size": 0.30,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 8,
    "class_weight": "balanced",
    "threshold": 0.50,
}


def load_data(archivo, hoja="Datos_Modelo"):
    """Carga la hoja de datos del Excel."""
    return pd.read_excel(archivo, sheet_name=hoja)


def build_pipeline(df):
    """Construye el pipeline de preprocesamiento + Random Forest."""
    columnas_predictoras = [c for c in df.columns if c not in EXCLUIR]

    X = df[columnas_predictoras].copy()

    columnas_numericas = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    columnas_categoricas = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    transformaciones = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                columnas_numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                columnas_categoricas,
            ),
        ],
        remainder="drop",
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=MODEL_CONFIG["n_estimators"],
        max_depth=MODEL_CONFIG["max_depth"],
        class_weight=MODEL_CONFIG["class_weight"],
        random_state=MODEL_CONFIG["random_state"],
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformaciones),
            ("modelo", modelo_rf),
        ]
    )

    return pipeline, X, df[TARGET].copy(), columnas_predictoras


def train_random_forest(df):
    """Entrena, evalúa y devuelve todos los resultados del modelo."""
    if TARGET not in df.columns:
        raise ValueError(
            f"No se encontró el target '{TARGET}' en la hoja Datos_Modelo."
        )

    pipeline, X, y, columnas_predictoras = build_pipeline(df)

    # 70% train / 30% test, estratificado, random_state 42
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    # Umbral autorizado = 0.50
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= MODEL_CONFIG["threshold"]).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    cm = confusion_matrix(y_test, y_pred)

    # Importancia de variables después del One-Hot Encoding
    preprocesador = pipeline.named_steps["preprocesamiento"]
    modelo = pipeline.named_steps["modelo"]

    nombres_variables = preprocesador.get_feature_names_out()
    importancias = modelo.feature_importances_

    importancia_df = (
        pd.DataFrame(
            {
                "Variable": nombres_variables,
                "Importancia": importancias,
            }
        )
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "confusion_matrix": cm,
        "importance": importancia_df,
        "columns": columnas_predictoras,
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }


def save_model(pipeline, path="random_forest_tension_liquidez.pkl"):
    """Guarda el pipeline entrenado."""
    import joblib
    joblib.dump(pipeline, path)


def importance_to_excel(importancia_df):
    """Devuelve las importancias en formato Excel en memoria."""
    buffer = io.BytesIO()
    importancia_df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


def plot_confusion_matrix(cm):
    """Crea la figura de matriz de confusión."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Matriz de confusión - Random Forest")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_xticks([0, 1], ["Sin tensión", "Tensión"])
    ax.set_yticks([0, 1], ["Sin tensión", "Tensión"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_roc(y_test, y_prob):
    """Crea la curva ROC."""
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("Curva ROC - Random Forest")
    fig.tight_layout()
    return fig


def plot_importance(importancia_df, top_n=15):
    """Crea el gráfico de las variables más importantes."""
    top = importancia_df.head(top_n).sort_values("Importancia")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Variable"], top["Importancia"])
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    ax.set_title(f"Top {top_n} variables - Random Forest")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Ejecución local opcional.
    archivo = "Base_Didactica_Random_Forest_Capital_Trabajo.xlsx"
    df = load_data(archivo)
    resultados = train_random_forest(df)

    print("=" * 60)
    print("RESULTADOS - RANDOM FOREST")
    print("=" * 60)
    for nombre, valor in resultados["metrics"].items():
        print(f"{nombre:10s}: {valor:.4f}")

    print("\nTop 15 variables:")
    print(resultados["importance"].head(15).to_string(index=False))

    save_model(resultados["pipeline"])
    resultados["importance"].to_excel(
        "importancia_variables_random_forest.xlsx",
        index=False,
    )
    print("\nModelo guardado.")
