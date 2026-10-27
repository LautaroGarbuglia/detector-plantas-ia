import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import sys


MODELO = Path("modelos/mejor_modelo_agua_v2.keras")


if len(sys.argv) < 2:
    print("\nUso:")
    print("python prediccion\\predecir_agua.py imagen.jpg")
    sys.exit()


RUTA_IMAGEN = Path(sys.argv[1])


if not RUTA_IMAGEN.exists():
    print(f"\nNo existe la imagen: {RUTA_IMAGEN}")
    sys.exit()


print("\nCargando modelo...")

model = tf.keras.models.load_model(
    MODELO
)


print("Cargando imagen...")

imagen = Image.open(
    RUTA_IMAGEN
).convert("RGB")


imagen = imagen.resize(
    (48, 48)
)


array = np.array(
    imagen,
    dtype=np.float32
)


array = array / 255.0


array = np.expand_dims(
    array,
    axis=0
)


print("Analizando...\n")


prediccion = model.predict(
    array,
    verbose=0
)[0]


clase = int(
    np.argmax(prediccion)
)


confianza = float(
    prediccion[clase] * 100
)


print("==============================")
print("RESULTADO")
print("==============================")

print(f"Clase detectada: {clase}")
print(f"Confianza: {confianza:.2f}%")

print("\nProbabilidades:")

for i, probabilidad in enumerate(prediccion):

    print(
        f"Clase {i}: "
        f"{probabilidad * 100:.2f}%"
    )