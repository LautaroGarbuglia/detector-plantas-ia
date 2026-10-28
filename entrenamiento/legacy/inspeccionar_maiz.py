import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# RUTAS
# ==========================================

RUTA = Path("datasets_originales/maiz_agua")

X = np.load(RUTA / "trainData.npy")
y = np.load(RUTA / "trainLabel.npy")


# ==========================================
# INFORMACIÓN DEL DATASET
# ==========================================

print("=" * 50)
print("INFORMACIÓN DEL DATASET")
print("=" * 50)

print("Forma imágenes:", X.shape)
print("Tipo:", X.dtype)

print("Valor mínimo:", X.min())
print("Valor máximo:", X.max())

print("\nEtiquetas encontradas:")

valores, cantidades = np.unique(
    y,
    return_counts=True
)

for etiqueta, cantidad in zip(
    valores,
    cantidades
):
    print(
        f"Etiqueta {etiqueta}: {cantidad} imágenes"
    )


# ==========================================
# CREAR CARPETA DE INSPECCIÓN
# ==========================================

SALIDA = Path(
    "inspeccion_dataset"
)

SALIDA.mkdir(
    exist_ok=True
)


# ==========================================
# NORMALIZAR IMAGEN PARA GUARDAR
# ==========================================

def preparar_imagen(img):

    img = np.array(img)

    # Si está normalizada entre 0 y 1
    if img.max() <= 1.0:
        img = img * 255

    img = np.clip(
        img,
        0,
        255
    )

    return img.astype(
        np.uint8
    )


# ==========================================
# GUARDAR EJEMPLOS
# ==========================================

CANTIDAD_EJEMPLOS = 15


for etiqueta in valores:

    carpeta = SALIDA / f"etiqueta_{etiqueta}"

    carpeta.mkdir(
        exist_ok=True
    )

    indices = np.where(
        y == etiqueta
    )[0]

    print(
        f"\nGuardando ejemplos de etiqueta {etiqueta}..."
    )

    for numero, indice in enumerate(
        indices[:CANTIDAD_EJEMPLOS]
    ):

        imagen = preparar_imagen(
            X[indice]
        )

        archivo = (
            carpeta /
            f"ejemplo_{numero + 1:02d}.jpg"
        )

        plt.imsave(
            archivo,
            imagen
        )


print("\n" + "=" * 50)
print("FINALIZADO")
print("=" * 50)

print(
    "\nRevisá la carpeta:"
)

print(
    "inspeccion_dataset"
)