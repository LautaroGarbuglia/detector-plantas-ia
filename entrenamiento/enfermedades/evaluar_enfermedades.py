"""
EVALUACIÓN DEL MODELO DE ENFERMEDADES.

Calcula:

- Accuracy.
- Matriz de confusión.
- Precision.
- Recall.
- F1 por clase.
"""

from pathlib import Path

import numpy as np
from tensorflow import keras


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DATASET_DIR = (

    PROJECT_ROOT
    / "datasets_originales"
    / "enfermedades"
    / "test"
)


MODEL_PATH = (

    Path(__file__)
    .resolve()
    .parent
    / "salidas"
    / "mejor_modelo_enfermedades.keras"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CLASS_NAMES = [

    "Blight",

    "Common_Rust",

    "Gray_Leaf_Spot",

    "Healthy"
]


IMG_SIZE = (
    160,
    160
)


BATCH_SIZE = 32


# ============================================================
# COMPROBAR ARCHIVOS
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"No existe: {DATASET_DIR}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"No existe: {MODEL_PATH}"
    )


# ============================================================
# DATASET
# ============================================================

test_ds = (
    keras.utils
    .image_dataset_from_directory(

        DATASET_DIR,

        class_names=CLASS_NAMES,

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False
    )
)


# ============================================================
# MODELO
# ============================================================

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ============================================================
# PREDICCIONES
# ============================================================

y_real = []

y_pred = []


for imagenes, etiquetas in test_ds:

    probabilidades = model.predict(

        imagenes,

        verbose=0
    )


    predicciones = np.argmax(

        probabilidades,

        axis=1
    )


    y_real.extend(
        etiquetas.numpy()
    )

    y_pred.extend(
        predicciones
    )


y_real = np.asarray(
    y_real
)

y_pred = np.asarray(
    y_pred
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

numero_clases = len(
    CLASS_NAMES
)


matriz = np.zeros(

    (
        numero_clases,
        numero_clases
    ),

    dtype=int
)


for real, predicho in zip(

    y_real,
    y_pred

):

    matriz[
        real,
        predicho
    ] += 1


# ============================================================
# ACCURACY
# ============================================================

accuracy = np.mean(
    y_real == y_pred
)


print()
print("=" * 70)
print("RESULTADOS - ENFERMEDADES")
print("=" * 70)

print()

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    "Matriz de confusión:"
)

print(
    matriz
)

print()


# ============================================================
# MÉTRICAS POR CLASE
# ============================================================

for i, clase in enumerate(
    CLASS_NAMES
):

    tp = matriz[i, i]

    fp = (
        matriz[:, i].sum()
        - tp
    )

    fn = (
        matriz[i, :].sum()
        - tp
    )


    precision = (

        tp / (tp + fp)

        if (tp + fp)

        else 0
    )


    recall = (

        tp / (tp + fn)

        if (tp + fn)

        else 0
    )


    f1 = (

        2
        * precision
        * recall
        / (precision + recall)

        if (
            precision
            + recall
        )

        else 0
    )


    print(
        clase
    )

    print(
        f"  Precision: "
        f"{precision:.4f}"
    )

    print(
        f"  Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"  F1:        "
        f"{f1:.4f}"
    )

    print()