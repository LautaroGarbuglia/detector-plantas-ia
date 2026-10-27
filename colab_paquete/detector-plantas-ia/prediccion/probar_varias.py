import numpy as np
import tensorflow as tf
from pathlib import Path

RUTA = Path("datasets_originales/maiz_agua")

X = np.load(RUTA / "testData.npy").astype("float32") / 255.0
y = np.load(RUTA / "testLabel.npy")

model = tf.keras.models.load_model(
    "modelos/mejor_modelo_agua_v2.keras"
)

cantidad = 20

predicciones = model.predict(
    X[:cantidad],
    verbose=0
)

clases = np.argmax(
    predicciones,
    axis=1
)

print("\nREAL | PREDICHA | CONFIANZA")
print("-" * 30)

aciertos = 0

for i in range(cantidad):

    confianza = predicciones[i][clases[i]] * 100

    correcto = "SI" if clases[i] == y[i] else "NO"

    if clases[i] == y[i]:
        aciertos += 1

    print(
        f"{y[i]}    | "
        f"{clases[i]}        | "
        f"{confianza:6.2f}% | "
        f"{correcto}"
    )

print("\nAciertos:")
print(f"{aciertos}/{cantidad}")