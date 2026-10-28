"""
============================================================
VERIFICACIÓN DE LOS MODELOS FINALES
============================================================

Este script NO entrena modelos.

Su función es comprobar los tres modelos que actualmente
utiliza la aplicación web:

    app/ia/agua.py
    app/ia/enfermedades.py
    app/ia/plagas.py

Verifica:

    - Dimensión de entrada.
    - Dimensión de salida.
    - Cantidad de parámetros.
    - Capas principales.
    - Capas Dense.
    - Dropout.
    - Rescaling.
    - MobileNetV2.
    - Clases.
    - Umbral del modelo de plagas.

Esto permite comparar los modelos desplegados con los
scripts existentes en entrenamiento/.
"""

from pathlib import Path
import sys

from tensorflow import keras


# ============================================================
# RAÍZ DEL PROYECTO
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


sys.path.insert(
    0,
    str(BASE_DIR)
)


# ============================================================
# IMPORTAR MODELOS DE LA APLICACIÓN
# ============================================================

from app.ia.agua import (
    cargar_modelo as cargar_agua,
    CLASES as CLASES_AGUA
)

from app.ia.enfermedades import (
    cargar_modelo as cargar_enfermedades,
    CLASES as CLASES_ENFERMEDADES
)

from app.ia.plagas import (
    cargar_modelo as cargar_plagas,
    CLASES as CLASES_PLAGAS,
    UMBRAL as UMBRAL_PLAGA
)


# ============================================================
# VALORES ESPERADOS
# ============================================================

MODELOS_ESPERADOS = {

    "AGUA": {

        "input_shape": (
            None,
            48,
            48,
            3
        ),

        "output_shape": (
            None,
            3
        ),

        "params": 406499
    },


    "ENFERMEDADES": {

        "input_shape": (
            None,
            160,
            160,
            3
        ),

        "output_shape": (
            None,
            4
        ),

        "params": 2422468
    },


    "PLAGAS": {

        "input_shape": (
            None,
            160,
            160,
            3
        ),

        "output_shape": (
            None,
            2
        ),

        "params": 2422210
    }
}


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def encabezado(texto):

    print()
    print("=" * 80)
    print(texto)
    print("=" * 80)


# ============================================================
# MOSTRAR CONFIGURACIÓN DE UNA CAPA
# ============================================================

def mostrar_capa(
    indice,
    layer
):

    nombre_tipo = (
        layer.__class__.__name__
    )


    print(
        f"{indice:03d} | "
        f"{layer.name:<35} | "
        f"{nombre_tipo}"
    )


    # --------------------------------------------------------
    # DENSE
    # --------------------------------------------------------

    if isinstance(
        layer,
        keras.layers.Dense
    ):

        print(
            f"      units: "
            f"{layer.units}"
        )

        print(
            f"      activation: "
            f"{layer.activation.__name__}"
        )


    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    if isinstance(
        layer,
        keras.layers.Dropout
    ):

        print(
            f"      rate: "
            f"{layer.rate}"
        )


    # --------------------------------------------------------
    # RESCALING
    # --------------------------------------------------------

    if isinstance(
        layer,
        keras.layers.Rescaling
    ):

        print(
            f"      scale: "
            f"{layer.scale}"
        )

        print(
            f"      offset: "
            f"{layer.offset}"
        )


    # --------------------------------------------------------
    # CONV2D
    # --------------------------------------------------------

    if isinstance(
        layer,
        keras.layers.Conv2D
    ):

        print(
            f"      filters: "
            f"{layer.filters}"
        )

        print(
            f"      kernel: "
            f"{layer.kernel_size}"
        )


# ============================================================
# BUSCAR CAPAS RECURSIVAMENTE
# ============================================================

def buscar_capas(
    modelo,
    tipo
):

    encontradas = []


    for layer in modelo.layers:

        if isinstance(
            layer,
            tipo
        ):

            encontradas.append(
                layer
            )


        # Algunos modelos, como MobileNetV2,
        # pueden aparecer como un modelo dentro de otro.

        if isinstance(
            layer,
            keras.Model
        ):

            encontradas.extend(

                buscar_capas(
                    layer,
                    tipo
                )
            )


    return encontradas


# ============================================================
# ANALIZAR MODELO
# ============================================================

def analizar_modelo(
    nombre,
    modelo,
    clases
):

    esperado = (
        MODELOS_ESPERADOS[
            nombre
        ]
    )


    encabezado(
        f"MODELO: {nombre}"
    )


    # ========================================================
    # INFORMACIÓN GENERAL
    # ========================================================

    print()

    print(
        "Nombre interno:",
        modelo.name
    )


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
        f"{modelo.count_params():,}"
    )


    print()

    print(
        "Clases:"
    )


    for indice, clase in enumerate(
        clases
    ):

        print(
            f"  {indice}: {clase}"
        )


    # ========================================================
    # VALIDACIONES
    # ========================================================

    print()
    print(
        "VERIFICACIONES:"
    )


    input_correcto = (
        tuple(
            modelo.input_shape
        )
        == esperado[
            "input_shape"
        ]
    )


    output_correcto = (
        tuple(
            modelo.output_shape
        )
        == esperado[
            "output_shape"
        ]
    )


    params_correctos = (
        modelo.count_params()
        == esperado[
            "params"
        ]
    )


    print(
        "  Input:",
        "✅" if input_correcto else "❌"
    )


    print(
        "  Output:",
        "✅" if output_correcto else "❌"
    )


    print(
        "  Parámetros:",
        "✅" if params_correctos else "❌"
    )


    # ========================================================
    # CAPAS PRINCIPALES
    # ========================================================

    print()
    print(
        "CAPAS PRINCIPALES:"
    )

    print()


    for indice, layer in enumerate(
        modelo.layers
    ):

        mostrar_capa(
            indice,
            layer
        )


    # ========================================================
    # DENSE
    # ========================================================

    dense_layers = buscar_capas(

        modelo,
        keras.layers.Dense
    )


    print()
    print(
        "CAPAS DENSE:"
    )


    for layer in dense_layers:

        print(
            f"  {layer.name}: "
            f"{layer.units} unidades"
        )


    # ========================================================
    # DROPOUT
    # ========================================================

    dropout_layers = buscar_capas(

        modelo,
        keras.layers.Dropout
    )


    print()
    print(
        "DROPOUT:"
    )


    if dropout_layers:

        for layer in dropout_layers:

            print(
                f"  {layer.name}: "
                f"{layer.rate}"
            )

    else:

        print(
            "  Ninguno"
        )


    # ========================================================
    # RESCALING
    # ========================================================

    rescaling_layers = buscar_capas(

        modelo,
        keras.layers.Rescaling
    )


    print()
    print(
        "RESCALING:"
    )


    if rescaling_layers:

        for layer in rescaling_layers:

            print(
                f"  {layer.name}: "
                f"scale={layer.scale}, "
                f"offset={layer.offset}"
            )

    else:

        print(
            "  No encontrado"
        )


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    pooling_layers = buscar_capas(

        modelo,
        keras.layers.GlobalAveragePooling2D
    )


    print()
    print(
        "GLOBAL AVERAGE POOLING:"
    )


    if pooling_layers:

        for layer in pooling_layers:

            print(
                f"  ✅ {layer.name}"
            )

    else:

        print(
            "  ❌ No encontrado"
        )


    # ========================================================
    # BUSCAR MOBILENET
    # ========================================================

    print()
    print(
        "MOBILENETV2:"
    )


    encontrado_mobilenet = False


    for layer in modelo.layers:

        nombre = (
            layer.name.lower()
        )


        if (
            "mobilenet" in nombre
        ):

            print(
                f"  ✅ {layer.name}"
            )

            encontrado_mobilenet = True


    if not encontrado_mobilenet:

        # El nombre puede no conservar literalmente
        # la palabra MobileNet.

        for layer in modelo.layers:

            if isinstance(
                layer,
                keras.Model
            ):

                print(
                    "  Modelo interno:",
                    layer.name
                )


    print()


# ============================================================
# CARGAR MODELOS
# ============================================================

encabezado(
    "CARGANDO MODELOS"
)


print()
print(
    "Cargando agua..."
)

modelo_agua = (
    cargar_agua()
)


print(
    "Cargando enfermedades..."
)

modelo_enfermedades = (
    cargar_enfermedades()
)


print(
    "Cargando plagas..."
)

modelo_plagas = (
    cargar_plagas()
)


# ============================================================
# ANALIZAR AGUA
# ============================================================

analizar_modelo(

    "AGUA",

    modelo_agua,

    CLASES_AGUA
)


# ============================================================
# ANALIZAR ENFERMEDADES
# ============================================================

analizar_modelo(

    "ENFERMEDADES",

    modelo_enfermedades,

    CLASES_ENFERMEDADES
)


# ============================================================
# ANALIZAR PLAGAS
# ============================================================

analizar_modelo(

    "PLAGAS",

    modelo_plagas,

    CLASES_PLAGAS
)


# ============================================================
# UMBRAL DE PLAGAS
# ============================================================

encabezado(
    "CONFIGURACIÓN ESPECIAL DE PLAGAS"
)


print()

print(
    "Umbral Fall Armyworm:",
    UMBRAL_PLAGA
)


print()

print(
    "Se considera Fall Armyworm cuando:"
)


print()

print(
    f"probabilidad_FAW >= "
    f"{UMBRAL_PLAGA}"
)


# ============================================================
# RESUMEN
# ============================================================

encabezado(
    "RESUMEN FINAL"
)


modelos = [

    (
        "AGUA",
        modelo_agua
    ),

    (
        "ENFERMEDADES",
        modelo_enfermedades
    ),

    (
        "PLAGAS",
        modelo_plagas
    )
]


todo_correcto = True


for nombre, modelo in modelos:

    esperado = (
        MODELOS_ESPERADOS[
            nombre
        ]
    )


    correcto = (

        tuple(
            modelo.input_shape
        )
        == esperado[
            "input_shape"
        ]

        and

        tuple(
            modelo.output_shape
        )
        == esperado[
            "output_shape"
        ]

        and

        modelo.count_params()
        == esperado[
            "params"
        ]
    )


    if not correcto:

        todo_correcto = False


    print(
        nombre,
        "✅ CORRECTO"
        if correcto
        else "❌ REVISAR"
    )


print()


if todo_correcto:

    print(
        "✅ Los tres modelos desplegados "
        "coinciden con las características "
        "esperadas."
    )


else:

    print(
        "⚠️ Hay diferencias que debemos revisar."
    )


print()
print("=" * 80)
print("VERIFICACIÓN FINALIZADA")
print("=" * 80)