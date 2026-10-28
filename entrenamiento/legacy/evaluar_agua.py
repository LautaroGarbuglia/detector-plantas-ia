import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================
# RUTAS
# ============================================

RUTA_DATOS = Path("datasets_originales/maiz_agua")
RUTA_MODELO = Path("modelos/mejor_modelo_agua.keras")


# ============================================
# CARGAR TEST
# ============================================

print("\nCargando conjunto de prueba...")

X_test = np.load(
    RUTA_DATOS / "testData.npy"
)

y_test = np.load(
    RUTA_DATOS / "testLabel.npy"
)


print("Imágenes:", X_test.shape)
print("Etiquetas:", y_test.shape)


# ============================================
# NORMALIZAR
# ============================================

X_test = X_test.astype("float32") / 255.0


# ============================================
# CARGAR MODELO
# ============================================

print("\nCargando modelo...")

model = tf.keras.models.load_model(
    RUTA_MODELO
)


# ============================================
# PREDECIR
# ============================================

print("\nRealizando predicciones...")

predicciones = model.predict(
    X_test,
    batch_size=32
)

y_pred = np.argmax(
    predicciones,
    axis=1
)


# ============================================
# ACCURACY
# ============================================

accuracy = np.mean(
    y_pred == y_test
)

print("\n====================================")
print("RESULTADOS")
print("====================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================
# REPORTE
# ============================================

print("\nReporte por clase:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Clase 0",
            "Clase 1",
            "Clase 2"
        ],
        digits=4
    )
)


# ============================================
# MATRIZ DE CONFUSIÓN
# ============================================

matriz = confusion_matrix(
    y_test,
    y_pred
)

print("\nMatriz de confusión:\n")
print(matriz)


# ============================================
# GRÁFICO
# ============================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=[
        "Clase 0",
        "Clase 1",
        "Clase 2"
    ]
)

disp.plot(
    cmap=None,
    values_format="d"
)

plt.title(
    "Matriz de confusión - Estrés hídrico"
)

plt.tight_layout()

plt.savefig(
    "matriz_confusion_agua.png",
    dpi=200
)

print(
    "\nImagen guardada:"
)

print(
    "matriz_confusion_agua.png"
)