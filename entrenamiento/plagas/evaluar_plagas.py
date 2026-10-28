"""
EVALUACIÓN DEL MODELO DE PLAGAS.

El modelo tiene dos clases:

0 - No Fall Armyworm
1 - Fall Armyworm

La aplicación final utiliza un umbral de 0.34
para decidir Fall Armyworm.
"""

from pathlib import Path

import numpy as np
from tensorflow import keras


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


TEST_DIR = (

    PROJECT_ROOT
    / "datasets_originales"
    / "plagas"
    / "test"
)


MODEL_PATH = (

    Path(__file__)
    .resolve()
    .parent
    / "salidas"
    / "mejor_modelo_plagas.keras"
)


CLASS_NAMES = [

    "No_Fall_Armyworm",

    "Fall_Armyworm"
]


IMG_SIZE = (
    160,
    160
)


BATCH_SIZE = 32


# Umbral elegido para la aplicación final.
UMBRAL = 0.34


# ============================================================
# VALIDAR
# ============================================================

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"No existe {TEST_DIR}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"No existe {MODEL_PATH}"
    )


# ============================================================
# DATASET
# ============================================================

test_ds = (
    keras.utils
    .image_dataset_from_directory(

        TEST_DIR,

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

prob_faw = []


for imagenes, etiquetas in test_ds:

    probabilidades = model.predict(

        imagenes,

        verbose=0
    )


    y_real.extend(
        etiquetas.numpy()
    )


    prob_faw.extend(

        probabilidades[
            :,
            1
        ]
    )


y_real = np.asarray(
    y_real
)


prob_faw = np.asarray(
    prob_faw
)


# ============================================================
# APLICAR UMBRAL
# ============================================================

y_pred = (

    prob_faw
    >= UMBRAL

).astype(int)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

tn = np.sum(
    (y_real == 0)
    & (y_pred == 0)
)


fp = np.sum(
    (y_real == 0)
    & (y_pred == 1)
)


fn = np.sum(
    (y_real == 1)
    & (y_pred == 0)
)


tp = np.sum(
    (y_real == 1)
    & (y_pred == 1)
)


# ============================================================
# MÉTRICAS
# ============================================================

accuracy = (

    (tp + tn)
    / len(y_real)
)


precision = (

    tp / (tp + fp)

    if (
        tp + fp
    )

    else 0
)


recall = (

    tp / (tp + fn)

    if (
        tp + fn
    )

    else 0
)


f1 = (

    2
    * precision
    * recall
    / (
        precision
        + recall
    )

    if (
        precision
        + recall
    )

    else 0
)


# ============================================================
# RESULTADOS
# ============================================================

print()
print("=" * 70)
print("RESULTADOS - FALL ARMYWORM")
print("=" * 70)

print(
    f"Umbral: {UMBRAL:.2f}"
)

print()

print(
    "Matriz de confusión:"
)

print()

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)

print()

print(
    f"Accuracy:  "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall:    "
    f"{recall * 100:.2f}%"
)

print(
    f"F1:        "
    f"{f1 * 100:.2f}%"
)