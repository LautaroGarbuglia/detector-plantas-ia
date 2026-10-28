"""
inspeccionar_modelos.py

Este script NO entrena los modelos.

Su objetivo es abrir los archivos .keras del proyecto y generar
documentación legible para poder entender su arquitectura desde
Visual Studio Code.

Genera para cada modelo:

- Un archivo TXT con:
    - función del modelo
    - tamaño de entrada
    - tamaño de salida
    - cantidad total de parámetros
    - capas utilizadas
    - model.summary()

- Un archivo JSON con la arquitectura completa de Keras.

Los pesos entrenados continúan almacenados dentro del archivo .keras.
"""

from pathlib import Path
import json
import tensorflow as tf


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELOS_DIR = BASE_DIR / "modelos"

DOCUMENTACION_DIR = (
    BASE_DIR / "documentacion_modelos"
)

DOCUMENTACION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INFORMACIÓN DE LOS MODELOS
# ============================================================

MODELOS = {

    "enfermedades": {

        "archivo":
            "mejor_modelo_enfermedades_maiz_v3.keras",

        "descripcion": (
            "Modelo utilizado para clasificar enfermedades "
            "visuales en hojas de maíz."
        ),

        "entrada":
            "Imagen RGB de 160 x 160 píxeles.",

        "salida": (
            "4 clases: Blight (Tizón), Common Rust "
            "(Roya común), Gray Leaf Spot (Mancha gris) "
            "y Healthy (Sana)."
        ),

        "preprocesamiento": (
            "El modelo contiene su propio preprocesamiento "
            "interno. app.py NO divide la imagen por 255."
        )
    },


    "agua": {

        "archivo":
            "modelo_agua_v2_colab_8478.keras",

        "descripcion": (
            "Modelo convolucional utilizado para clasificar "
            "patrones visuales relacionados con tratamientos "
            "hídricos en hojas de maíz."
        ),

        "entrada":
            "Imagen RGB de 48 x 48 píxeles.",

        "salida":
            "3 clases: Clase 0, Clase 1 y Clase 2.",

        "preprocesamiento": (
            "Antes de ingresar al modelo, app.py convierte "
            "los píxeles a float32 y divide sus valores por 255."
        )
    },


    "plagas": {

        "archivo":
            "mejor_modelo_plagas_maiz_v3.keras",

        "descripcion": (
            "Modelo binario utilizado para detectar evidencia "
            "visual de Fall Armyworm o gusano cogollero."
        ),

        "entrada":
            "Imagen RGB de 160 x 160 píxeles.",

        "salida": (
            "2 clases: No Fall Armyworm y Fall Armyworm."
        ),

        "preprocesamiento": (
            "El modelo contiene una capa de reescalado interna. "
            "app.py NO divide la imagen por 255."
        ),

        "decision": (
            "La aplicación utiliza un umbral de 0.34 para "
            "decidir si existe evidencia suficiente de "
            "Fall Armyworm."
        )
    }
}


# ============================================================
# FUNCIÓN PARA INSPECCIONAR UN MODELO
# ============================================================

def inspeccionar_modelo(
    nombre,
    informacion
):

    ruta_modelo = (
        MODELOS_DIR /
        informacion["archivo"]
    )


    print()
    print("=" * 70)
    print(f"MODELO: {nombre.upper()}")
    print("=" * 70)


    # --------------------------------------------------------
    # COMPROBAR QUE EL ARCHIVO EXISTA
    # --------------------------------------------------------

    if not ruta_modelo.exists():

        print(
            f"ERROR: no existe {ruta_modelo}"
        )

        return


    # --------------------------------------------------------
    # CARGAR EL MODELO
    # --------------------------------------------------------

    print(
        f"Cargando: {ruta_modelo.name}"
    )

    modelo = tf.keras.models.load_model(
        ruta_modelo,
        compile=False
    )


    # --------------------------------------------------------
    # OBTENER MODEL.SUMMARY()
    # --------------------------------------------------------

    lineas_summary = []

    modelo.summary(
        print_fn=lineas_summary.append
    )


    # --------------------------------------------------------
    # INFORMACIÓN GENERAL
    # --------------------------------------------------------

    lineas = []

    lineas.append(
        f"MODELO: {nombre.upper()}"
    )

    lineas.append(
        "=" * 70
    )

    lineas.append("")

    lineas.append(
        "ARCHIVO:"
    )

    lineas.append(
        informacion["archivo"]
    )

    lineas.append("")


    lineas.append(
        "FUNCIÓN:"
    )

    lineas.append(
        informacion["descripcion"]
    )

    lineas.append("")


    lineas.append(
        "ENTRADA ESPERADA:"
    )

    lineas.append(
        informacion["entrada"]
    )

    lineas.append("")


    lineas.append(
        "SALIDA:"
    )

    lineas.append(
        informacion["salida"]
    )

    lineas.append("")


    lineas.append(
        "PREPROCESAMIENTO:"
    )

    lineas.append(
        informacion["preprocesamiento"]
    )

    lineas.append("")


    if "decision" in informacion:

        lineas.append(
            "REGLA DE DECISIÓN:"
        )

        lineas.append(
            informacion["decision"]
        )

        lineas.append("")


    # --------------------------------------------------------
    # FORMAS REALES DEL MODELO
    # --------------------------------------------------------

    lineas.append(
        "INPUT SHAPE:"
    )

    lineas.append(
        str(modelo.input_shape)
    )

    lineas.append("")


    lineas.append(
        "OUTPUT SHAPE:"
    )

    lineas.append(
        str(modelo.output_shape)
    )

    lineas.append("")


    lineas.append(
        "PARÁMETROS TOTALES:"
    )

    lineas.append(
        f"{modelo.count_params():,}"
    )

    lineas.append("")


    # --------------------------------------------------------
    # LISTADO DE CAPAS
    # --------------------------------------------------------

    lineas.append(
        "=" * 70
    )

    lineas.append(
        "CAPAS DEL MODELO"
    )

    lineas.append(
        "=" * 70
    )

    lineas.append("")


    for numero, capa in enumerate(
        modelo.layers
    ):

        lineas.append(
            f"{numero:03d} | "
            f"{capa.name} | "
            f"{capa.__class__.__name__} | "
            f"trainable={capa.trainable}"
        )


    lineas.append("")
    lineas.append("")


    # --------------------------------------------------------
    # SUMMARY COMPLETO
    # --------------------------------------------------------

    lineas.append(
        "=" * 70
    )

    lineas.append(
        "MODEL.SUMMARY()"
    )

    lineas.append(
        "=" * 70
    )

    lineas.append("")

    lineas.extend(
        lineas_summary
    )


    # --------------------------------------------------------
    # GUARDAR TXT
    # --------------------------------------------------------

    archivo_txt = (
        DOCUMENTACION_DIR /
        f"{nombre}_resumen.txt"
    )


    archivo_txt.write_text(
        "\n".join(lineas),
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # GUARDAR ARQUITECTURA JSON
    # --------------------------------------------------------

    arquitectura = json.loads(
        modelo.to_json()
    )


    archivo_json = (
        DOCUMENTACION_DIR /
        f"{nombre}_arquitectura.json"
    )


    archivo_json.write_text(

        json.dumps(
            arquitectura,
            indent=4,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


    print(
        f"TXT generado: {archivo_txt.name}"
    )

    print(
        f"JSON generado: {archivo_json.name}"
    )

    print(
        f"Input: {modelo.input_shape}"
    )

    print(
        f"Output: {modelo.output_shape}"
    )

    print(
        f"Parámetros: {modelo.count_params():,}"
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print()
    print(
        "INSPECCIÓN DE MODELOS DEL "
        "DETECTOR DE MAÍZ"
    )


    for nombre, informacion in (
        MODELOS.items()
    ):

        inspeccionar_modelo(
            nombre,
            informacion
        )


    print()
    print("=" * 70)

    print(
        "Documentación generada correctamente."
    )

    print(
        f"Carpeta: {DOCUMENTACION_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()