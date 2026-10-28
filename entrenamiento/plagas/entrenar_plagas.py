"""
============================================================
ENTRENAMIENTO DEL MODELO DE PLAGAS DEL MAÍZ
============================================================

Modelo encargado de detectar Fall Armyworm
(gusano cogollero).

ARQUITECTURA VERIFICADA:

    Entrada:
        160 x 160 x 3

    Data Augmentation
        ↓
    Rescaling [-1, 1]
        ↓
    MobileNetV2
        ↓
    GlobalAveragePooling2D
        ↓
    Dense 128 ReLU
        ↓
    Dropout 0.35
        ↓
    Dense 2 Softmax

Parámetros:
    2.422.210

CLASES:

    0 -> No Fall Armyworm
    1 -> Fall Armyworm

UMBRAL UTILIZADO POR LA WEB:

    0.34

La arquitectura está verificada contra
app/ia/plagas.py.

La estrategia exacta histórica de entrenamiento
no se conserva, por lo que transfer learning,
fine tuning y augmentation se documentan como
una reconstrucción reproducible.
"""

from pathlib import Path

import tensorflow as tf
import keras
from keras import layers


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 123

IMG_SIZE = (
    160,
    160
)

BATCH_SIZE = 32

EPOCHS_TRANSFER = 15

EPOCHS_FINE_TUNING = 10

NUM_CLASES = 2


CLASES = [

    "No_Fall_Armyworm",

    "Fall_Armyworm"
]


UMBRAL = 0.34


tf.random.set_seed(
    SEED
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)


RUTA_DATASET = (
    BASE_DIR
    / "datasets_originales"
    / "plagas"
)


RUTA_TRAIN = (
    RUTA_DATASET
    / "train"
)


RUTA_VALIDACION = (
    RUTA_DATASET
    / "validacion"
)


RUTA_TEST = (
    RUTA_DATASET
    / "test"
)


RUTA_SALIDA = (
    Path(__file__)
    .resolve()
    .parent
    / "salidas"
)


RUTA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)


RUTA_MEJOR_MODELO = (
    RUTA_SALIDA
    / "mejor_modelo_plagas.keras"
)


RUTA_MODELO_FINAL = (
    RUTA_SALIDA
    / "modelo_plagas_final.keras"
)


RUTA_PESOS = (
    RUTA_SALIDA
    / "plagas.weights.h5"
)


# ============================================================
# COMPROBAR DATASET
# ============================================================

print()
print("=" * 70)
print("COMPROBANDO DATASET DE PLAGAS")
print("=" * 70)


for ruta in [

    RUTA_TRAIN,
    RUTA_VALIDACION,
    RUTA_TEST

]:

    if not ruta.exists():

        raise FileNotFoundError(

            "\nNo se encontró:\n"
            f"{ruta}\n\n"
            "Todavía necesitamos recuperar "
            "o agregar el dataset de plagas."
        )


# ============================================================
# DATASETS
# ============================================================

train_ds = (
    keras.utils
    .image_dataset_from_directory(

        RUTA_TRAIN,

        class_names=CLASES,

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=True,

        seed=SEED
    )
)


val_ds = (
    keras.utils
    .image_dataset_from_directory(

        RUTA_VALIDACION,

        class_names=CLASES,

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False
    )
)


test_ds = (
    keras.utils
    .image_dataset_from_directory(

        RUTA_TEST,

        class_names=CLASES,

        image_size=IMG_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False
    )
)


# ============================================================
# PREFETCH
# ============================================================

AUTOTUNE = (
    tf.data.AUTOTUNE
)


train_ds = train_ds.prefetch(
    AUTOTUNE
)


val_ds = val_ds.prefetch(
    AUTOTUNE
)


test_ds = test_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )

    ],

    name="data_augmentation"
)


# ============================================================
# MOBILENETV2
# ============================================================

base_model = (
    keras.applications.MobileNetV2(

        input_shape=(
            160,
            160,
            3
        ),

        include_top=False,

        weights="imagenet"
    )
)


base_model.trainable = False


# ============================================================
# ENTRADA
# ============================================================

inputs = keras.Input(

    shape=(
        160,
        160,
        3
    ),

    name="input_layer_1"
)


# ============================================================
# AUGMENTATION
# ============================================================

x = data_augmentation(
    inputs
)


# ============================================================
# NORMALIZACIÓN MOBILENET
# ============================================================

x = layers.Rescaling(

    scale=1.0 / 127.5,

    offset=-1,

    name="rescaling"

)(x)


# ============================================================
# MOBILENET
# ============================================================

x = base_model(

    x,

    training=False
)


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

x = layers.GlobalAveragePooling2D(

    name="global_average_pooling2d"

)(x)


# ============================================================
# DENSE
# ============================================================

x = layers.Dense(

    128,

    activation="relu",

    name="dense"

)(x)


# ============================================================
# DROPOUT
# ============================================================

x = layers.Dropout(

    0.35,

    name="dropout"

)(x)


# ============================================================
# SALIDA
# ============================================================

outputs = layers.Dense(

    NUM_CLASES,

    activation="softmax",

    name="dense_1"

)(x)


# ============================================================
# MODELO
# ============================================================

model = keras.Model(

    inputs=inputs,

    outputs=outputs
)


# ============================================================
# RESUMEN
# ============================================================

model.summary()


print()

print(
    "Parámetros:",
    f"{model.count_params():,}"
)


# ============================================================
# VERIFICAR ARQUITECTURA
# ============================================================

PARAMETROS_ESPERADOS = (
    2422210
)


if (
    model.count_params()
    != PARAMETROS_ESPERADOS
):

    raise ValueError(

        "\nLa arquitectura no coincide "
        "con el modelo final.\n"
        f"Esperados: {PARAMETROS_ESPERADOS:,}\n"
        f"Actuales: {model.count_params():,}"
    )


print()

print(
    "✅ Arquitectura verificada:"
)

print(
    "2.422.210 parámetros."
)


# ============================================================
# COMPILAR
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(

        filepath=RUTA_MEJOR_MODELO,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),


    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1
    ),


    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1
    )
]


# ============================================================
# TRANSFER LEARNING
# ============================================================

print()
print("=" * 70)
print("FASE 1 - TRANSFER LEARNING")
print("=" * 70)


model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_TRANSFER,

    callbacks=callbacks
)


# ============================================================
# FINE TUNING
# ============================================================

print()
print("=" * 70)
print("FASE 2 - FINE TUNING")
print("=" * 70)


base_model.trainable = True


for layer in (
    base_model.layers[:-30]
):

    layer.trainable = False


for layer in (
    base_model.layers
):

    if isinstance(

        layer,

        layers.BatchNormalization

    ):

        layer.trainable = False


# ============================================================
# RECOMPILAR
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# ENTRENAR FINE TUNING
# ============================================================

model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_FINE_TUNING,

    callbacks=callbacks
)


# ============================================================
# MEJOR MODELO
# ============================================================

model = keras.models.load_model(
    RUTA_MEJOR_MODELO
)


# ============================================================
# TEST CON ARGMAX
# ============================================================

loss, accuracy = model.evaluate(

    test_ds,

    verbose=1
)


print()
print(
    f"Accuracy estándar: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# EVALUACIÓN CON UMBRAL 0.34
# ============================================================

print()
print("=" * 70)
print("EVALUACIÓN CON UMBRAL 0.34")
print("=" * 70)


reales = []

probabilidades_faw = []


for imagenes, etiquetas in test_ds:

    predicciones = model.predict(

        imagenes,

        verbose=0
    )


    reales.extend(
        etiquetas.numpy()
    )


    probabilidades_faw.extend(
        predicciones[:, 1]
    )


import numpy as np


reales = np.asarray(
    reales
)


probabilidades_faw = np.asarray(
    probabilidades_faw
)


predicciones_umbral = (

    probabilidades_faw
    >= UMBRAL

).astype(
    int
)


accuracy_umbral = np.mean(

    predicciones_umbral
    == reales
)


print()

print(
    "Umbral:",
    UMBRAL
)


print(
    f"Accuracy con umbral: "
    f"{accuracy_umbral * 100:.2f}%"
)


# ============================================================
# GUARDAR
# ============================================================

model.save(
    RUTA_MODELO_FINAL
)


model.save_weights(
    RUTA_PESOS
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 70)

print()

print(
    "Modelo:"
)

print(
    RUTA_MODELO_FINAL
)


print()

print(
    "Pesos:"
)

print(
    RUTA_PESOS
)


print()

print(
    "Umbral Fall Armyworm:",
    UMBRAL
)