"""
============================================================
ENTRENAMIENTO DEL MODELO DE ENFERMEDADES DEL MAÍZ
============================================================

Este script reconstruye el pipeline de entrenamiento
correspondiente a la arquitectura del modelo final
utilizado actualmente por la aplicación.

MODELO DESPLEGADO:

    Entrada:
        160 x 160 x 3

    Arquitectura:
        Data Augmentation
        Rescaling [-1, 1]
        MobileNetV2
        GlobalAveragePooling2D
        Dense 128 ReLU
        Dropout 0.35
        Dense 4 Softmax

    Parámetros:
        2.422.468

CLASES:

    0 -> Blight
    1 -> Common Rust
    2 -> Gray Leaf Spot
    3 -> Healthy

IMPORTANTE:

La arquitectura está verificada contra el modelo
desplegado en app/ia/enfermedades.py.

La estrategia concreta de entrenamiento y las
transformaciones de aumento de datos fueron reconstruidas
porque el script histórico final no se conserva.
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


NUM_CLASES = 4


CLASES = [

    "Blight",

    "Common_Rust",

    "Gray_Leaf_Spot",

    "Healthy"
]


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
    / "enfermedades"
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
    / "mejor_modelo_enfermedades.keras"
)


RUTA_MODELO_FINAL = (
    RUTA_SALIDA
    / "modelo_enfermedades_final.keras"
)


RUTA_PESOS = (
    RUTA_SALIDA
    / "enfermedades.weights.h5"
)


# ============================================================
# COMPROBAR DATASET
# ============================================================

print()
print("=" * 70)
print("COMPROBANDO DATASET DE ENFERMEDADES")
print("=" * 70)


for ruta in [

    RUTA_TRAIN,
    RUTA_VALIDACION,
    RUTA_TEST

]:

    if not ruta.exists():

        raise FileNotFoundError(

            "\nNo se encontró el dataset requerido:\n"
            f"{ruta}\n\n"
            "Todavía necesitamos recuperar o agregar "
            "el dataset de enfermedades."
        )


# ============================================================
# CARGAR DATASET
# ============================================================

print()
print("=" * 70)
print("CARGANDO DATASET")
print("=" * 70)


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
# OPTIMIZACIÓN DEL PIPELINE
# ============================================================

AUTOTUNE = (
    tf.data.AUTOTUNE
)


train_ds = (
    train_ds
    .prefetch(
        AUTOTUNE
    )
)


val_ds = (
    val_ds
    .prefetch(
        AUTOTUNE
    )
)


test_ds = (
    test_ds
    .prefetch(
        AUTOTUNE
    )
)


# ============================================================
# DATA AUGMENTATION
# ============================================================
#
# El modelo final contiene una capa Sequential antes
# del Rescaling.
#
# Las transformaciones concretas se reconstruyen porque
# el entrenamiento histórico final no está disponible.
#

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

    name="sequential"
)


# ============================================================
# MOBILENETV2
# ============================================================

print()
print("=" * 70)
print("CREANDO MOBILENETV2")
print("=" * 70)


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


# ============================================================
# TRANSFER LEARNING
# ============================================================

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
# DATA AUGMENTATION
# ============================================================

x = data_augmentation(
    inputs
)


# ============================================================
# RESCALING
# ============================================================
#
# Convierte:
#
#       0 ... 255
#
# a:
#
#      -1 ... 1
#
# Esto está dentro del modelo final.
#

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
# DENSE 128
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
# CLASIFICADOR
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

    outputs=outputs,

    name="enfermedades_maiz_v3"
)


# ============================================================
# MOSTRAR MODELO
# ============================================================

print()

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
    2422468
)


if (
    model.count_params()
    != PARAMETROS_ESPERADOS
):

    raise ValueError(

        "\nLa arquitectura creada no coincide "
        "con el modelo final.\n"
        f"Esperados: {PARAMETROS_ESPERADOS:,}\n"
        f"Actuales: {model.count_params():,}"
    )


print()
print(
    "✅ Arquitectura verificada:"
)

print(
    "2.422.468 parámetros."
)


# ============================================================
# COMPILAR FASE 1
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
# FASE 1
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
# FASE 2 - FINE TUNING
# ============================================================

print()
print("=" * 70)
print("FASE 2 - FINE TUNING")
print("=" * 70)


base_model.trainable = True


# Dejamos congelada la mayor parte
# de MobileNetV2.

for layer in (
    base_model.layers[:-30]
):

    layer.trainable = False


# BatchNormalization permanece congelado.

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
# FINE TUNING
# ============================================================

model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_FINE_TUNING,

    callbacks=callbacks
)


# ============================================================
# CARGAR MEJOR MODELO
# ============================================================

model = keras.models.load_model(
    RUTA_MEJOR_MODELO
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("EVALUACIÓN FINAL")
print("=" * 70)


loss, accuracy = model.evaluate(

    test_ds,

    verbose=1
)


print()

print(
    f"Test loss: "
    f"{loss:.4f}"
)


print(
    f"Test accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# GUARDAR MODELO
# ============================================================

model.save(
    RUTA_MODELO_FINAL
)


# ============================================================
# GUARDAR PESOS
# ============================================================

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