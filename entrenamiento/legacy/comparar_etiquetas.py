import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RUTA = Path("datasets_originales/maiz_agua")

X = np.load(RUTA / "trainData.npy")
y = np.load(RUTA / "trainLabel.npy")


ETIQUETAS = [0, 1, 2]

EJEMPLOS = 8


fig, axes = plt.subplots(
    3,
    EJEMPLOS,
    figsize=(16, 7)
)


for fila, etiqueta in enumerate(ETIQUETAS):

    indices = np.where(y == etiqueta)[0]

    for columna in range(EJEMPLOS):

        indice = indices[columna]

        imagen = X[indice]

        if imagen.max() <= 1:
            imagen = imagen * 255

        imagen = np.clip(
            imagen,
            0,
            255
        ).astype("uint8")

        axes[fila, columna].imshow(imagen)

        axes[fila, columna].axis("off")

        if columna == 0:
            axes[fila, columna].set_title(
                f"Etiqueta {etiqueta}",
                fontsize=12
            )


plt.tight_layout()

plt.savefig(
    "comparacion_etiquetas.png",
    dpi=200
)

print(
    "Imagen creada: comparacion_etiquetas.png"
)