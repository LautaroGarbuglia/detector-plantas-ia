import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


# ==========================================
# CONFIGURACIÓN
# ==========================================

RUTA = Path("datasets_originales/maiz_agua")

BATCH_SIZE = 32
EPOCHS = 40

tf.random.set_seed(123)
np.random.seed(123)


# ==========================================
# CARGAR DATASET
# ==========================================

print("\n======================================")
print("CARGANDO DATASET")
print("======================================\n")

X_train = np.load(RUTA / "trainData.npy")
y_train = np.load(RUTA / "trainLabel.npy")

X_val = np.load(RUTA / "valData.npy")
y_val = np.load(RUTA / "valLabel.npy")

X_test = np.load(RUTA / "testData.npy")
y_test = np.load(RUTA / "testLabel.npy")


print("Train:", X_train.shape)
print("Validación:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# NORMALIZACIÓN
# ==========================================

X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# ==========================================
# DATA AUGMENTATION
# ==========================================

augmentation = keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.05
    )

], name="data_augmentation")


# ==========================================
# MODELO
# ==========================================

inputs = keras.Input(
    shape=(48, 48, 3)
)

x = augmentation(inputs)


# BLOQUE 1

x = layers.Conv2D(
    32,
    3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.Activation(
    "relu"
)(x)

x = layers.MaxPooling2D()(x)


# BLOQUE 2

x = layers.Conv2D(
    64,
    3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.Activation(
    "relu"
)(x)

x = layers.MaxPooling2D()(x)


# BLOQUE 3

x = layers.Conv2D(
    128,
    3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.Activation(
    "relu"
)(x)

x = layers.MaxPooling2D()(x)


# BLOQUE 4

x = layers.Conv2D(
    256,
    3,
    padding="same",
    use_bias=False
)(x)

x = layers.BatchNormalization()(x)

x = layers.Activation(
    "relu"
)(x)


# ==========================================
# CLASIFICADOR
# ==========================================

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.35
)(x)

x = layers.Dense(
    64,
    activation="relu"
)(x)

x = layers.Dropout(
    0.25
)(x)


outputs = layers.Dense(
    3,
    activation="softmax"
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="modelo_agua_v2"
)


# ==========================================
# COMPILAR
# ==========================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()


# ==========================================
# CALLBACKS
# ==========================================

Path("modelos").mkdir(
    exist_ok=True
)


callbacks = [

    keras.callbacks.ModelCheckpoint(

        "modelos/mejor_modelo_agua_v2.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=0.00001,

        verbose=1
    )

]


# ==========================================
# ENTRENAMIENTO
# ==========================================

print("\n======================================")
print("ENTRENANDO MODELO V2")
print("======================================\n")


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks

)


# ==========================================
# CARGAR EL MEJOR MODELO
# ==========================================

print("\nCargando mejor modelo...")

mejor_modelo = keras.models.load_model(
    "modelos/mejor_modelo_agua_v2.keras"
)


# ==========================================
# EVALUACIÓN
# ==========================================

print("\n======================================")
print("RESULTADO FINAL")
print("======================================\n")


loss, accuracy = mejor_modelo.evaluate(
    X_test,
    y_test
)


print(
    f"\nAccuracy TEST V2: "
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# GUARDAR
# ==========================================

mejor_modelo.save(
    "modelos/modelo_agua_v2.keras"
)


print(
    "\nModelo guardado:"
)

print(
    "modelos/modelo_agua_v2.keras"
)