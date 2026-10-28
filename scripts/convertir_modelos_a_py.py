"""
convertir_modelos_a_py.py

Convierte los modelos .keras actuales en:

    arquitectura/modelo -> archivo .py
    pesos entrenados    -> archivo .weights.h5

IMPORTANTE:
Este script NO vuelve a entrenar los modelos.

También verifica automáticamente que el modelo original
y el modelo reconstruido produzcan prácticamente la misma salida.
"""

from pathlib import Path
import importlib.util

import numpy as np
import tensorflow as tf


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELOS_DIR = BASE_DIR / "modelos"

MODELOS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODELOS ACTUALES
# ============================================================

MODELOS = {

    "enfermedades": {

        "archivo":
            "mejor_modelo_enfermedades_maiz_v3.keras",

        "descripcion":
            "Clasificación de enfermedades en hojas de maíz.",

        "clases": [
            "Blight (Tizón)",
            "Common Rust (Roya común)",
            "Gray Leaf Spot (Mancha gris)",
            "Healthy (Sana)"
        ],

        "preprocesamiento": (
            "Imagen RGB de 160x160. "
            "El modelo posee preprocesamiento interno."
        )
    },


    "agua": {

        "archivo":
            "modelo_agua_v2_colab_8478.keras",

        "descripcion":
            "Clasificación experimental de patrones hídricos.",

        "clases": [
            "Clase 0",
            "Clase 1",
            "Clase 2"
        ],

        "preprocesamiento": (
            "Imagen RGB de 48x48. "
            "Antes de ingresar al modelo se divide por 255."
        )
    },


    "plagas": {

        "archivo":
            "mejor_modelo_plagas_maiz_v3.keras",

        "descripcion": (
            "Clasificación binaria para detectar "
            "Fall Armyworm (gusano cogollero)."
        ),

        "clases": [
            "No Fall Armyworm",
            "Fall Armyworm"
        ],

        "preprocesamiento": (
            "Imagen RGB de 160x160. "
            "El modelo posee preprocesamiento interno."
        ),

        "umbral":
            0.34
    }
}


# ============================================================
# CREAR ARCHIVO PYTHON DE UN MODELO
# ============================================================

def crear_archivo_python(
    nombre,
    informacion,
    modelo
):

    arquitectura_json = modelo.to_json()

    archivo_pesos = (
        f"{nombre}.weights.h5"
    )


    # ========================================================
    # INFORMACIÓN DE CAPAS PARA COMENTARIOS
    # ========================================================

    comentarios_capas = []

    for numero, capa in enumerate(
        modelo.layers
    ):

        try:

            forma_salida = str(
                capa.output.shape
            )

        except Exception:

            forma_salida = (
                "No disponible"
            )


        comentarios_capas.append(

            f"# {numero:02d}. "
            f"{capa.name} | "
            f"{capa.__class__.__name__} | "
            f"Salida: {forma_salida}"
        )


    bloque_capas = "\n".join(
        comentarios_capas
    )


    clases_repr = repr(
        informacion["clases"]
    )


    descripcion_repr = repr(
        informacion["descripcion"]
    )


    preprocesamiento_repr = repr(
        informacion[
            "preprocesamiento"
        ]
    )


    # ========================================================
    # UMBRAL OPCIONAL
    # ========================================================

    if "umbral" in informacion:

        codigo_umbral = (
            f"\nUMBRAL = "
            f"{informacion['umbral']}\n"
        )

    else:

        codigo_umbral = ""


    # ========================================================
    # CÓDIGO DEL ARCHIVO .PY
    # ========================================================

    codigo = f'''"""
MODELO: {nombre.upper()}

Este archivo representa la arquitectura del modelo entrenado.

NO entrena el modelo.

Los pesos aprendidos se encuentran en:

    {archivo_pesos}

Función:
    {informacion["descripcion"]}

Entrada:
    {modelo.input_shape}

Salida:
    {modelo.output_shape}

Parámetros:
    {modelo.count_params():,}

Preprocesamiento:
    {informacion["preprocesamiento"]}
"""

from pathlib import Path
import tensorflow as tf


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

DESCRIPCION = {descripcion_repr}

CLASES = {clases_repr}

PREPROCESAMIENTO = {preprocesamiento_repr}

INPUT_SHAPE = {repr(modelo.input_shape)}

OUTPUT_SHAPE = {repr(modelo.output_shape)}

PARAMETROS = {modelo.count_params()}
{codigo_umbral}

# ============================================================
# CAPAS PRINCIPALES DEL MODELO
# ============================================================
#
# La siguiente lista permite entender visualmente las capas
# principales sin tener que interpretar el archivo de pesos.
#
{bloque_capas}


# ============================================================
# ARQUITECTURA SERIALIZADA
# ============================================================
#
# Keras guarda internamente una configuración completa del modelo.
#
# Se conserva aquí para reconstruir EXACTAMENTE la arquitectura
# original, incluyendo modelos complejos como MobileNetV2.
#
# Esto evita tener que reconstruir manualmente cientos de capas.
#

MODEL_JSON = {repr(arquitectura_json)}


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo():

    """
    Reconstruye únicamente la arquitectura.

    En este punto todavía NO se cargaron
    los pesos entrenados.
    """

    modelo = (
        tf.keras.models.model_from_json(
            MODEL_JSON
        )
    )

    return modelo


# ============================================================
# CARGAR MODELO ENTRENADO
# ============================================================

def cargar_modelo(
    ruta_pesos=None
):

    """
    Crea la arquitectura y carga
    los pesos obtenidos durante el entrenamiento.

    Devuelve el modelo listo para usar.
    """

    modelo = crear_modelo()


    if ruta_pesos is None:

        ruta_pesos = (
            Path(__file__)
            .resolve()
            .parent
            / "{archivo_pesos}"
        )


    modelo.load_weights(
        ruta_pesos
    )


    return modelo


# ============================================================
# INSPECCIÓN MANUAL
# ============================================================

if __name__ == "__main__":

    modelo = cargar_modelo()

    print()
    print("=" * 70)
    print("MODELO: {nombre.upper()}")
    print("=" * 70)

    print(
        "Entrada:",
        modelo.input_shape
    )

    print(
        "Salida:",
        modelo.output_shape
    )

    print(
        "Parámetros:",
        f"{{modelo.count_params():,}}"
    )

    print()

    modelo.summary()
'''


    ruta_python = (
        MODELOS_DIR /
        f"{nombre}.py"
    )


    ruta_python.write_text(
        codigo,
        encoding="utf-8"
    )


    return ruta_python


# ============================================================
# IMPORTAR EL NUEVO ARCHIVO PYTHON
# ============================================================

def importar_modulo(
    nombre,
    ruta
):

    spec = (
        importlib.util.spec_from_file_location(
            f"modelo_{nombre}",
            ruta
        )
    )


    modulo = (
        importlib.util.module_from_spec(
            spec
        )
    )


    spec.loader.exec_module(
        modulo
    )


    return modulo


# ============================================================
# COMPARAR MODELO ORIGINAL VS MODELO NUEVO
# ============================================================

def verificar_modelos(
    nombre,
    original,
    nuevo
):

    forma = original.input_shape


    dimensiones = [
        1 if valor is None
        else valor

        for valor in forma
    ]


    rng = np.random.default_rng(
        12345
    )


    # --------------------------------------------------------
    # Agua recibe datos aproximadamente entre 0 y 1.
    # Enfermedades y plagas reciben valores RGB 0-255.
    # --------------------------------------------------------

    if nombre == "agua":

        entrada = rng.random(
            dimensiones
        ).astype(
            np.float32
        )

    else:

        entrada = rng.uniform(
            0,
            255,
            dimensiones
        ).astype(
            np.float32
        )


    salida_original = original.predict(
        entrada,
        verbose=0
    )


    salida_nueva = nuevo.predict(
        entrada,
        verbose=0
    )


    diferencia = float(
        np.max(
            np.abs(
                salida_original
                - salida_nueva
            )
        )
    )


    return diferencia


# ============================================================
# CONVERTIR TODOS
# ============================================================

def convertir():

    # Necesario para importar:
    #
    # from modelos.agua import ...
    #
    init_file = (
        MODELOS_DIR /
        "__init__.py"
    )


    init_file.touch(
        exist_ok=True
    )


    print()
    print("=" * 70)
    print("CONVERSIÓN DE MODELOS")
    print("=" * 70)


    for nombre, informacion in (
        MODELOS.items()
    ):

        print()
        print(
            f"Convirtiendo: "
            f"{nombre.upper()}"
        )


        ruta_original = (
            MODELOS_DIR /
            informacion["archivo"]
        )


        if not ruta_original.exists():

            print(
                "ERROR: no existe:"
            )

            print(
                ruta_original
            )

            continue


        # ----------------------------------------------------
        # CARGAR MODELO ORIGINAL
        # ----------------------------------------------------

        original = (
            tf.keras.models.load_model(
                ruta_original,
                compile=False
            )
        )


        # ----------------------------------------------------
        # GUARDAR PESOS
        # ----------------------------------------------------

        ruta_pesos = (
            MODELOS_DIR /
            f"{nombre}.weights.h5"
        )


        original.save_weights(
            ruta_pesos
        )


        print(
            "Pesos:",
            ruta_pesos.name
        )


        # ----------------------------------------------------
        # GENERAR .PY
        # ----------------------------------------------------

        ruta_python = (
            crear_archivo_python(
                nombre,
                informacion,
                original
            )
        )


        print(
            "Código:",
            ruta_python.name
        )


        # ----------------------------------------------------
        # CARGAR DESDE EL NUEVO .PY
        # ----------------------------------------------------

        modulo = importar_modulo(
            nombre,
            ruta_python
        )


        reconstruido = (
            modulo.cargar_modelo()
        )


        # ----------------------------------------------------
        # VERIFICAR
        # ----------------------------------------------------

        diferencia = verificar_modelos(
            nombre,
            original,
            reconstruido
        )


        print(
            "Diferencia máxima:",
            diferencia
        )


        if diferencia <= 1e-6:

            print(
                "✅ MODELO VERIFICADO"
            )

        else:

            print(
                "⚠️ REVISAR MODELO"
            )


    print()
    print("=" * 70)
    print(
        "CONVERSIÓN FINALIZADA"
    )
    print("=" * 70)


if __name__ == "__main__":

    convertir()