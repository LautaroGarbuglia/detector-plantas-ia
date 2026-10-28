import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


# ============================================
# CONFIGURACIÓN
# ============================================

RUTA = Path("datasets_originales/maiz_agua")

BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASES = 3


# ============================================
# CARGAR DATOS
# ============================================

print("\nCargando dataset...")

X_train = np.load(RUTA / "trainData.npy")
y_train = np.load(RUTA / "trainLabel.npy")

X_val = np.load(RUTA / "valData.npy")
y_val = np.load(RUTA / "valLabel.npy")

X_test = np.load(RUTA / "testData.npy")
y_test = np.load(RUTA / "testLabel.npy")


print("Entrenamiento:", X_train.shape)
print("Validación:", X_val.shape)
print("Prueba:", X_test.shape)


# ============================================
# NORMALIZAR IMÁGENES
# ============================================

X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# ============================================
# DATA AUGMENTATION
# ============================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])


# ============================================
# CREAR MODELO CNN
# ============================================

model = keras.Sequential([

    layers.Input(shape=(48, 48, 3)),

    data_augmentation,

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        NUM_CLASES,
        activation="softmax"
    )
])


# ============================================
# COMPILAR
# ============================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# ============================================
# CALLBACKS
# ============================================

Path("modelos").mkdir(
    exist_ok=True
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        "modelos/mejor_modelo_agua.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

]


# ============================================
# ENTRENAR
# ============================================

print("\n====================================")
print("INICIANDO ENTRENAMIENTO")
print("====================================\n")


history = model.fit(

    X_train,
    y_train,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=callbacks
)


# ============================================
# EVALUAR CON TEST
# ============================================

print("\n====================================")
print("EVALUACIÓN FINAL")
print("====================================\n")


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)


print(
    f"\nAccuracy en TEST: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================
# GUARDAR MODELO
# ============================================

model.save(
    "modelos/modelo_agua.keras"
)


print("\nModelo guardado correctamente.")