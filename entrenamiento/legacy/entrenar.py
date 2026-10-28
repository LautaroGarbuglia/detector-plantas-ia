import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path


# ========================================
# CONFIGURACIÓN
# ========================================

RUTA_DATASET = Path("dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 123


# ========================================
# COMPROBAR DATASET
# ========================================

if not RUTA_DATASET.exists():
    raise FileNotFoundError(
        "No se encontró la carpeta dataset."
    )

print("\n========================================")
print("CARGANDO DATASET")
print("========================================\n")


# ========================================
# DATASET DE ENTRENAMIENTO
# ========================================

train_ds = keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ========================================
# DATASET DE VALIDACIÓN
# ========================================

val_ds = keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ========================================
# NOMBRES DE LAS CLASES
# ========================================

class_names = train_ds.class_names

print("\nClases detectadas:")

for indice, clase in enumerate(class_names):
    print(f"{indice}: {clase}")

print()


# ========================================
# OPTIMIZAR DATASET
# ========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ========================================
# DATA AUGMENTATION
# ========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


# ========================================
# MODELO BASE MOBILENETV2
# ========================================

print("\n========================================")
print("CARGANDO MOBILENETV2")
print("========================================\n")


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Congelamos las capas preentrenadas
base_model.trainable = False


# ========================================
# CREAR RED NEURONAL
# ========================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# ========================================
# COMPILAR
# ========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ========================================
# MOSTRAR MODELO
# ========================================

model.summary()


# ========================================
# CALLBACKS
# ========================================

Path("modelos").mkdir(
    exist_ok=True
)

callbacks = [

    keras.callbacks.ModelCheckpoint(
        "modelos/mejor_modelo.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]


# ========================================
# ENTRENAR
# ========================================

print("\n========================================")
print("INICIANDO ENTRENAMIENTO")
print("========================================\n")


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ========================================
# GUARDAR MODELO FINAL
# ========================================

model.save(
    "modelos/modelo_plantas.keras"
)


print("\n========================================")
print("ENTRENAMIENTO FINALIZADO")
print("========================================")

print(
    "Modelo guardado en:"
)

print(
    "modelos/modelo_plantas.keras"
)

print(
    "Mejor modelo guardado en:"
)

print(
    "modelos/mejor_modelo.keras"
)