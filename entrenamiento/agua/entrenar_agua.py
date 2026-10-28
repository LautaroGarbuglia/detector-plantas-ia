"""
============================================================
ENTRENAMIENTO DEL MODELO DE ESTRÉS HÍDRICO EN MAÍZ
============================================================

Este es el pipeline final organizado del modelo de agua.

Está basado en la versión V2 conservada del proyecto y en
la arquitectura del modelo final utilizado por la aplicación.

IMPORTANTE:
    - Entrada: imágenes RGB de 48x48 píxeles.
    - Salida: 3 clases.
    - Normalización: píxeles / 255.0.
    - Arquitectura CNN entrenada desde cero.
    - No utiliza MobileNetV2.

CLASES:
    0 -> Clase 0
    1 -> Clase 1
    2 -> Clase 2

Los nombres reales de los tratamientos hídricos todavía
no fueron verificados documentalmente, por eso no se
inventan nombres para las clases.

ARQUITECTURA:

    Input 48x48x3
          ↓
    Data Augmentation
          ↓
    Conv2D 32
    BatchNormalization
    ReLU
    MaxPooling
          ↓
    Conv2D 64
    BatchNormalization
    ReLU
    MaxPooling
          ↓
    Conv2D 128
    BatchNormalization
    ReLU
    MaxPooling
          ↓
    Conv2D 256
    BatchNormalization
    ReLU
          ↓
    GlobalAveragePooling2D
          ↓
    Dropout 0.35
          ↓
    Dense 64 ReLU
          ↓
    Dropout 0.25
          ↓
    Dense 3 Softmax

Parámetros esperados:
    406.499
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

SEED = 123

IMG_HEIGHT = 48
IMG_WIDTH = 48
CHANNELS = 3

NUM_CLASES = 3

BATCH_SIZE = 32

EPOCHS = 40


# ============================================================
# SEMILLAS
# ============================================================
#
# Ayudan a que el entrenamiento sea más reproducible.
#

np.random.seed(
    SEED
)

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


RUTA_DATOS = (
    BASE_DIR
    / "datasets_originales"
    / "maiz_agua"
    / "MaizeLeaf_dataset"
)


SALIDA_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "salidas"
)


SALIDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RUTA_MEJOR_MODELO = (
    SALIDA_DIR
    / "mejor_modelo_agua.keras"
)


RUTA_MODELO_FINAL = (
    SALIDA_DIR
    / "modelo_agua_final.keras"
)


RUTA_PESOS = (
    SALIDA_DIR
    / "agua.weights.h5"
)


# ============================================================
# ARCHIVOS DEL DATASET
# ============================================================

ARCHIVOS = {

    "X_train":
        RUTA_DATOS
        / "trainData.npy",

    "y_train":
        RUTA_DATOS
        / "trainLabel.npy",

    "X_val":
        RUTA_DATOS
        / "valData.npy",

    "y_val":
        RUTA_DATOS
        / "valLabel.npy",

    "X_test":
        RUTA_DATOS
        / "testData.npy",

    "y_test":
        RUTA_DATOS
        / "testLabel.npy"
}


# ============================================================
# COMPROBAR DATASET
# ============================================================

print()
print("=" * 70)
print("COMPROBANDO DATASET")
print("=" * 70)


for nombre, ruta in ARCHIVOS.items():

    if not ruta.exists():

        raise FileNotFoundError(
            "\nNo se encontró el archivo:\n"
            f"{ruta}\n"
        )

    print(
        f"✅ {nombre}: {ruta.name}"
    )


# ============================================================
# CARGAR DATASET
# ============================================================

print()
print("=" * 70)
print("CARGANDO DATASET")
print("=" * 70)


X_train = np.load(
    ARCHIVOS["X_train"]
)

y_train = np.load(
    ARCHIVOS["y_train"]
)


X_val = np.load(
    ARCHIVOS["X_val"]
)

y_val = np.load(
    ARCHIVOS["y_val"]
)


X_test = np.load(
    ARCHIVOS["X_test"]
)

y_test = np.load(
    ARCHIVOS["y_test"]
)


# ============================================================
# MOSTRAR FORMAS
# ============================================================

print()
print(
    "Train imágenes:",
    X_train.shape
)

print(
    "Train etiquetas:",
    y_train.shape
)


print(
    "Validación imágenes:",
    X_val.shape
)

print(
    "Validación etiquetas:",
    y_val.shape
)


print(
    "Test imágenes:",
    X_test.shape
)

print(
    "Test etiquetas:",
    y_test.shape
)


# ============================================================
# VALIDAR FORMATO
# ============================================================

forma_esperada = (
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS
)


if X_train.shape[1:] != forma_esperada:

    raise ValueError(
        "El conjunto train no tiene "
        f"la forma esperada {forma_esperada}. "
        f"Forma actual: {X_train.shape[1:]}"
    )


if X_val.shape[1:] != forma_esperada:

    raise ValueError(
        "El conjunto de validación no tiene "
        f"la forma esperada {forma_esperada}. "
        f"Forma actual: {X_val.shape[1:]}"
    )


if X_test.shape[1:] != forma_esperada:

    raise ValueError(
        "El conjunto test no tiene "
        f"la forma esperada {forma_esperada}. "
        f"Forma actual: {X_test.shape[1:]}"
    )


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

def mostrar_distribucion(
    nombre,
    etiquetas
):

    print()
    print(
        f"Distribución {nombre}:"
    )

    valores, cantidades = np.unique(
        etiquetas,
        return_counts=True
    )

    for valor, cantidad in zip(
        valores,
        cantidades
    ):

        print(
            f"  Clase {int(valor)}: "
            f"{int(cantidad)}"
        )


mostrar_distribucion(
    "TRAIN",
    y_train
)

mostrar_distribucion(
    "VALIDACIÓN",
    y_val
)

mostrar_distribucion(
    "TEST",
    y_test
)


# ============================================================
# NORMALIZAR
# ============================================================
#
# El modelo de agua fue entrenado con valores entre 0 y 1.
#
# Los valores originales de las imágenes se convierten
# de:
#
#       0 - 255
#
# a:
#
#       0.0 - 1.0
#
# mediante división por 255.
#

print()
print("=" * 70)
print("NORMALIZANDO IMÁGENES")
print("=" * 70)


X_train = (
    X_train.astype(
        "float32"
    )
    / 255.0
)


X_val = (
    X_val.astype(
        "float32"
    )
    / 255.0
)


X_test = (
    X_test.astype(
        "float32"
    )
    / 255.0
)


# ============================================================
# DATA AUGMENTATION
# ============================================================
#
# Genera pequeñas variaciones durante el entrenamiento.
#
# Las transformaciones aleatorias solamente actúan cuando
# training=True.
#
# No cambian las etiquetas.
#

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.08
        ),

        layers.RandomContrast(
            0.08
        )

    ],

    name="data_augmentation"
)


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo():

    """
    Construye la CNN utilizada para clasificar
    los tres patrones hídricos.

    Devuelve:
        keras.Model
    """

    # --------------------------------------------------------
    # ENTRADA
    # --------------------------------------------------------

    inputs = keras.Input(

        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS
        ),

        name="input_layer"
    )


    # --------------------------------------------------------
    # AUMENTO DE DATOS
    # --------------------------------------------------------

    x = data_augmentation(
        inputs
    )


    # ========================================================
    # BLOQUE CONVOLUCIONAL 1
    # ========================================================
    #
    # Aprende características básicas:
    #
    # - bordes
    # - colores
    # - pequeñas texturas
    #

    x = layers.Conv2D(

        filters=32,

        kernel_size=3,

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Activation(
        "relu"
    )(x)


    x = layers.MaxPooling2D()(x)


    # ========================================================
    # BLOQUE CONVOLUCIONAL 2
    # ========================================================

    x = layers.Conv2D(

        filters=64,

        kernel_size=3,

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Activation(
        "relu"
    )(x)


    x = layers.MaxPooling2D()(x)


    # ========================================================
    # BLOQUE CONVOLUCIONAL 3
    # ========================================================

    x = layers.Conv2D(

        filters=128,

        kernel_size=3,

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Activation(
        "relu"
    )(x)


    x = layers.MaxPooling2D()(x)


    # ========================================================
    # BLOQUE CONVOLUCIONAL 4
    # ========================================================
    #
    # Extrae características visuales más complejas.
    #

    x = layers.Conv2D(

        filters=256,

        kernel_size=3,

        padding="same",

        use_bias=False

    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.Activation(
        "relu"
    )(x)


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================
    #
    # Convierte los mapas de características en un vector
    # de 256 valores.
    #
    # Es más eficiente que Flatten para esta arquitectura.
    #

    x = layers.GlobalAveragePooling2D()(x)


    # ========================================================
    # REGULARIZACIÓN
    # ========================================================

    x = layers.Dropout(
        0.35
    )(x)


    # ========================================================
    # CAPA DENSA
    # ========================================================

    x = layers.Dense(

        64,

        activation="relu"

    )(x)


    # ========================================================
    # SEGUNDA REGULARIZACIÓN
    # ========================================================

    x = layers.Dropout(
        0.25
    )(x)


    # ========================================================
    # SALIDA
    # ========================================================
    #
    # Softmax genera tres probabilidades.
    #
    # Ejemplo:
    #
    # Clase 0 -> 0.75
    # Clase 1 -> 0.20
    # Clase 2 -> 0.05
    #

    outputs = layers.Dense(

        NUM_CLASES,

        activation="softmax"

    )(x)


    # ========================================================
    # MODELO FINAL
    # ========================================================

    modelo = keras.Model(

        inputs=inputs,

        outputs=outputs,

        name="modelo_agua_v2"
    )


    return modelo


# ============================================================
# CONSTRUIR MODELO
# ============================================================

print()
print("=" * 70)
print("CREANDO MODELO CNN")
print("=" * 70)


model = crear_modelo()


# ============================================================
# RESUMEN
# ============================================================

model.summary()


print()
print(
    "Parámetros totales:",
    f"{model.count_params():,}"
)


# ============================================================
# VERIFICAR CANTIDAD DE PARÁMETROS
# ============================================================

PARAMETROS_ESPERADOS = 406499


if model.count_params() == PARAMETROS_ESPERADOS:

    print()
    print(
        "✅ Arquitectura verificada:"
    )

    print(
        "406.499 parámetros."
    )


else:

    print()
    print(
        "⚠️ ADVERTENCIA:"
    )

    print(
        "La arquitectura no coincide con "
        "la cantidad esperada."
    )

    print(
        "Esperados:",
        PARAMETROS_ESPERADOS
    )

    print(
        "Actuales:",
        model.count_params()
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

    # --------------------------------------------------------
    # Guardar solamente la mejor versión.
    # --------------------------------------------------------

    keras.callbacks.ModelCheckpoint(

        filepath=RUTA_MEJOR_MODELO,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),


    # --------------------------------------------------------
    # Detener el entrenamiento si deja de mejorar.
    # --------------------------------------------------------

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True,

        verbose=1
    ),


    # --------------------------------------------------------
    # Reducir learning rate automáticamente.
    # --------------------------------------------------------

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-6,

        verbose=1
    )
]


# ============================================================
# ENTRENAMIENTO
# ============================================================

print()
print("=" * 70)
print("INICIANDO ENTRENAMIENTO")
print("=" * 70)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    shuffle=True
)


# ============================================================
# CARGAR MEJOR MODELO
# ============================================================

print()
print("=" * 70)
print("CARGANDO MEJOR MODELO")
print("=" * 70)


mejor_modelo = (
    keras.models.load_model(
        RUTA_MEJOR_MODELO
    )
)


# ============================================================
# EVALUACIÓN SOBRE TEST
# ============================================================

print()
print("=" * 70)
print("EVALUACIÓN SOBRE TEST")
print("=" * 70)


loss, accuracy = (
    mejor_modelo.evaluate(

        X_test,

        y_test,

        batch_size=BATCH_SIZE,

        verbose=1
    )
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
# GUARDAR MODELO FINAL
# ============================================================

mejor_modelo.save(
    RUTA_MODELO_FINAL
)


# ============================================================
# GUARDAR SOLAMENTE LOS PESOS
# ============================================================

mejor_modelo.save_weights(
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
    "Mejor modelo:"
)

print(
    RUTA_MEJOR_MODELO
)


print()

print(
    "Modelo final:"
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