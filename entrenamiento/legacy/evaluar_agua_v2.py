import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


RUTA_DATOS = Path("datasets_originales/maiz_agua")
RUTA_MODELO = Path("modelos/mejor_modelo_agua_v2.keras")


print("\nCargando conjunto de prueba...")

X_test = np.load(RUTA_DATOS / "testData.npy")
y_test = np.load(RUTA_DATOS / "testLabel.npy")

X_test = X_test.astype("float32") / 255.0


print("Cargando modelo V2...")

model = tf.keras.models.load_model(
    RUTA_MODELO
)


print("Realizando predicciones...")

predicciones = model.predict(
    X_test,
    batch_size=32
)

y_pred = np.argmax(
    predicciones,
    axis=1
)


accuracy = np.mean(
    y_pred == y_test
)


print("\n====================================")
print("RESULTADOS MODELO V2")
print("====================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


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


matriz = confusion_matrix(
    y_test,
    y_pred
)

print("\nMatriz de confusión:\n")
print(matriz)


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
    "Matriz de confusión - Modelo agua V2"
)

plt.tight_layout()

plt.savefig(
    "matriz_confusion_agua_v2.png",
    dpi=200
)

print(
    "\nImagen guardada: matriz_confusion_agua_v2.png"
)