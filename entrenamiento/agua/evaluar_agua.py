"""
============================================================
EVALUACIÓN DEL MODELO FINAL DE AGUA
============================================================

Este script evalúa EXACTAMENTE el modelo desplegado en
la aplicación Flask:

    app/ia/agua.py
    app/ia/agua.weights.h5

No vuelve a entrenar.

Utiliza:

    datasets_originales/
        maiz_agua/
            MaizeLeaf_dataset/
                testData.npy
                testLabel.npy

Calcula:

    - Accuracy global.
    - Matriz de confusión.
    - Precision por clase.
    - Recall por clase.
    - F1 por clase.
    - Distribución del conjunto test.

Las clases se mantienen como:

    Clase 0
    Clase 1
    Clase 2

porque todavía no se comprobó documentalmente
qué tratamiento hídrico representa cada número.
"""

from pathlib import Path

import sys

import numpy as np


# ============================================================
# RAÍZ DEL PROYECTO
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# HACER IMPORTABLE app/
# ============================================================

sys.path.insert(
    0,
    str(BASE_DIR)
)


# ============================================================
# IMPORTAR MODELO UTILIZADO POR FLASK
# ============================================================

from app.ia.agua import (
    cargar_modelo,
    CLASES
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

NUM_CLASES = 3

BATCH_SIZE = 32

PARAMETROS_ESPERADOS = 406499


# ============================================================
# RUTA DEL DATASET
# ============================================================

RUTA_DATOS = (

    BASE_DIR
    / "datasets_originales"
    / "maiz_agua"
    / "MaizeLeaf_dataset"
)


RUTA_X_TEST = (
    RUTA_DATOS
    / "testData.npy"
)


RUTA_Y_TEST = (
    RUTA_DATOS
    / "testLabel.npy"
)


# ============================================================
# COMPROBAR ARCHIVOS
# ============================================================

print()
print("=" * 70)
print("COMPROBANDO ARCHIVOS")
print("=" * 70)


if not RUTA_X_TEST.exists():

    raise FileNotFoundError(
        "\nNo se encontró:\n"
        f"{RUTA_X_TEST}"
    )


if not RUTA_Y_TEST.exists():

    raise FileNotFoundError(
        "\nNo se encontró:\n"
        f"{RUTA_Y_TEST}"
    )


print(
    "✅ testData.npy"
)

print(
    "✅ testLabel.npy"
)


# ============================================================
# CARGAR TEST
# ============================================================

print()
print("=" * 70)
print("CARGANDO CONJUNTO TEST")
print("=" * 70)


X_test = np.load(
    RUTA_X_TEST
)


y_test = np.load(
    RUTA_Y_TEST
)


print()

print(
    "Imágenes:",
    X_test.shape
)


print(
    "Etiquetas:",
    y_test.shape
)


# ============================================================
# COMPROBAR DIMENSIONES
# ============================================================

forma_esperada = (
    48,
    48,
    3
)


if X_test.shape[1:] != forma_esperada:

    raise ValueError(

        "El dataset no tiene "
        "la dimensión esperada.\n"
        f"Esperada: {forma_esperada}\n"
        f"Actual: {X_test.shape[1:]}"
    )


# ============================================================
# NORMALIZAR
# ============================================================
#
# El modelo final espera imágenes float32
# normalizadas entre 0 y 1.
#

print()
print(
    "Normalizando imágenes..."
)


X_test = (

    X_test.astype(
        "float32"
    )

    / 255.0
)


# ============================================================
# CARGAR MODELO FINAL
# ============================================================

print()
print("=" * 70)
print("CARGANDO MODELO FINAL")
print("=" * 70)


modelo = cargar_modelo()


print()

print(
    "Input shape:",
    modelo.input_shape
)


print(
    "Output shape:",
    modelo.output_shape
)


print(
    "Parámetros:",
    f"{modelo.count_params():,}"
)


# ============================================================
# VERIFICAR PARÁMETROS
# ============================================================

if (
    modelo.count_params()
    == PARAMETROS_ESPERADOS
):

    print()

    print(
        "✅ Arquitectura correcta."
    )

    print(
        "El modelo tiene exactamente "
        "406.499 parámetros."
    )


else:

    print()

    print(
        "⚠️ ADVERTENCIA"
    )

    print(
        "Se esperaban:",
        PARAMETROS_ESPERADOS
    )

    print(
        "Se encontraron:",
        modelo.count_params()
    )


# ============================================================
# PREDICCIONES
# ============================================================

print()
print("=" * 70)
print("REALIZANDO PREDICCIONES")
print("=" * 70)


probabilidades = modelo.predict(

    X_test,

    batch_size=BATCH_SIZE,

    verbose=1
)


# ============================================================
# CLASE PREDICHA
# ============================================================

y_pred = np.argmax(

    probabilidades,

    axis=1
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = float(

    np.mean(
        y_pred == y_test
    )
)


# ============================================================
# CREAR MATRIZ DE CONFUSIÓN MANUALMENTE
# ============================================================

matriz = np.zeros(

    (
        NUM_CLASES,
        NUM_CLASES
    ),

    dtype=int
)


for real, predicha in zip(

    y_test,
    y_pred

):

    matriz[
        int(real),
        int(predicha)
    ] += 1


# ============================================================
# RESULTADOS GENERALES
# ============================================================

print()
print("=" * 70)
print("RESULTADOS MODELO DE AGUA")
print("=" * 70)

print()

print(
    f"Accuracy test: "
    f"{accuracy * 100:.2f}%"
)


print()

print(
    "Matriz de confusión:"
)

print()

print(
    matriz
)


# ============================================================
# MÉTRICAS POR CLASE
# ============================================================

print()
print("=" * 70)
print("MÉTRICAS POR CLASE")
print("=" * 70)


for indice in range(
    NUM_CLASES
):

    # --------------------------------------------------------
    # TRUE POSITIVES
    # --------------------------------------------------------

    tp = matriz[
        indice,
        indice
    ]


    # --------------------------------------------------------
    # FALSE POSITIVES
    # --------------------------------------------------------

    fp = (

        matriz[:, indice].sum()

        - tp
    )


    # --------------------------------------------------------
    # FALSE NEGATIVES
    # --------------------------------------------------------

    fn = (

        matriz[indice, :].sum()

        - tp
    )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            / (tp + fp)
        )

    else:

        precision = 0.0


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            / (tp + fn)
        )

    else:

        recall = 0.0


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (
        precision + recall
    ) > 0:

        f1 = (

            2
            * precision
            * recall

            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    # --------------------------------------------------------
    # NOMBRE
    # --------------------------------------------------------

    if indice < len(
        CLASES
    ):

        nombre = (
            CLASES[indice]
        )

    else:

        nombre = (
            f"Clase {indice}"
        )


    # --------------------------------------------------------
    # MOSTRAR
    # --------------------------------------------------------

    print()

    print(
        nombre
    )


    print(
        f"  Precision: "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )


    print(
        f"  Recall:    "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )


    print(
        f"  F1:        "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )


# ============================================================
# DISTRIBUCIÓN DEL DATASET
# ============================================================

print()
print("=" * 70)
print("DISTRIBUCIÓN DEL CONJUNTO TEST")
print("=" * 70)


valores, cantidades = np.unique(

    y_test,

    return_counts=True
)


total = len(
    y_test
)


for valor, cantidad in zip(

    valores,
    cantidades

):

    porcentaje = (

        cantidad
        / total
        * 100
    )


    print()

    print(
        f"Clase {int(valor)}:"
    )

    print(
        f"  Imágenes: "
        f"{int(cantidad)}"
    )

    print(
        f"  Porcentaje: "
        f"{porcentaje:.2f}%"
    )


# ============================================================
# PREDICCIONES POR CLASE
# ============================================================

print()
print("=" * 70)
print("PREDICCIONES GENERADAS")
print("=" * 70)


valores_pred, cantidades_pred = (
    np.unique(

        y_pred,

        return_counts=True
    )
)


for valor, cantidad in zip(

    valores_pred,
    cantidades_pred

):

    print(
        f"Clase {int(valor)}: "
        f"{int(cantidad)} predicciones"
    )


# ============================================================
# CANTIDAD DE ACIERTOS
# ============================================================

aciertos = int(

    np.sum(
        y_pred == y_test
    )
)


errores = int(

    np.sum(
        y_pred != y_test
    )
)


print()
print("=" * 70)
print("RESUMEN")
print("=" * 70)

print()

print(
    "Total imágenes:",
    total
)


print(
    "Aciertos:",
    aciertos
)


print(
    "Errores:",
    errores
)


print(
    f"Accuracy:",
    f"{accuracy * 100:.2f}%"
)


print()

print("=" * 70)
print("EVALUACIÓN FINALIZADA")
print("=" * 70)