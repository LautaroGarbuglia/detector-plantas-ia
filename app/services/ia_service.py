"""
SERVICE - Inteligencia Artificial

Este servicio conecta la aplicación web
con los tres modelos entrenados.

Modelos utilizados:

1. Enfermedades.
2. Estrés hídrico.
3. Gusano cogollero.

El controlador solamente llama:

    analizar_imagen(ruta)

y este servicio se encarga del resto.
"""

from PIL import Image

import numpy as np


# ============================================================
# IMPORTAR MODELOS DE IA
# ============================================================

from app.ia.enfermedades import (

    cargar_modelo
    as cargar_modelo_enfermedades,

    CLASES
    as CLASES_ENFERMEDADES
)


from app.ia.agua import (

    cargar_modelo
    as cargar_modelo_agua,

    CLASES
    as CLASES_AGUA
)


from app.ia.plagas import (

    cargar_modelo
    as cargar_modelo_plagas,

    UMBRAL
    as UMBRAL_PLAGA
)


# ============================================================
# CARGAR MODELOS
# ============================================================

print(
    "Cargando modelo de enfermedades..."
)

modelo_enfermedades = (
    cargar_modelo_enfermedades()
)


print(
    "Cargando modelo de estrés hídrico..."
)

modelo_agua = (
    cargar_modelo_agua()
)


print(
    "Cargando modelo de plagas..."
)

modelo_plagas = (
    cargar_modelo_plagas()
)


print(
    "✅ Modelos de IA cargados correctamente."
)


# ============================================================
# INFORMACIÓN DE ENFERMEDADES
# ============================================================

INFO_ENFERMEDADES = {

    "Blight (Tizón)": {

        "titulo":
            "Tizón foliar",

        "descripcion": (
            "El modelo encontró características visuales "
            "similares a las hojas clasificadas como tizón."
        ),

        "sintomas": (
            "Pueden aparecer lesiones alargadas, zonas "
            "marrones y tejido seco sobre la hoja."
        ),

        "recomendacion": (
            "Revisar otras hojas y observar si aparecen "
            "lesiones similares o nuevas zonas secas."
        )
    },


    "Common Rust (Roya común)": {

        "titulo":
            "Roya común",

        "descripcion": (
            "La imagen presenta características similares "
            "a las hojas utilizadas para entrenar "
            "la clase Roya común."
        ),

        "sintomas": (
            "Pueden observarse pequeñas pústulas "
            "marrones, rojizas o anaranjadas."
        ),

        "recomendacion": (
            "Revisar ambas caras de otras hojas "
            "para comprobar si existen lesiones similares."
        )
    },


    "Gray Leaf Spot (Mancha gris)": {

        "titulo":
            "Mancha gris",

        "descripcion": (
            "El modelo encontró características "
            "compatibles con mancha gris."
        ),

        "sintomas": (
            "Suelen aparecer lesiones alargadas "
            "o rectangulares de color gris o marrón."
        ),

        "recomendacion": (
            "Revisar otras hojas y buscar "
            "manchas rectangulares similares."
        )
    },


    "Healthy (Sana)": {

        "titulo":
            "Hoja aparentemente sana",

        "descripcion": (
            "El modelo encontró mayor similitud "
            "con las hojas sanas de entrenamiento."
        ),

        "sintomas": (
            "No se detectaron características dominantes "
            "de las enfermedades conocidas por el modelo."
        ),

        "recomendacion": (
            "Continuar observando la planta. "
            "El resultado no descarta otros problemas "
            "que el modelo no conozca."
        )
    }
}


# ============================================================
# INFORMACIÓN DE PLAGAS
# ============================================================

INFO_PLAGA_POSITIVA = {

    "titulo":
        "Posible presencia de gusano cogollero",

    "descripcion": (
        "El modelo encontró características visuales "
        "compatibles con Fall Armyworm."
    ),

    "sintomas": (
        "Revisar perforaciones, tejido raspado, "
        "bordes dañados y daños cerca del cogollo."
    ),

    "recomendacion": (
        "Inspeccionar manualmente la planta "
        "y especialmente la zona del cogollo."
    )
}


INFO_PLAGA_NEGATIVA = {

    "titulo":
        "Sin evidencia suficiente de gusano cogollero",

    "descripcion": (
        "La probabilidad calculada está por debajo "
        "del umbral utilizado para considerar "
        "la presencia de Fall Armyworm."
    ),

    "sintomas": (
        "El modelo no encontró suficientes características "
        "similares a sus ejemplos de gusano cogollero."
    ),

    "recomendacion": (
        "Esto no descarta otras plagas. "
        "Si existen daños visibles, revisar "
        "la planta manualmente."
    )
}


# ============================================================
# NIVEL ORIENTATIVO
# ============================================================

def nivel_confianza(
    probabilidad
):

    if probabilidad >= 0.85:

        return "Alta"

    elif probabilidad >= 0.65:

        return "Media"

    else:

        return "Baja"


# ============================================================
# ANALIZAR IMAGEN
# ============================================================

def analizar_imagen(
    ruta
):

    imagen = Image.open(
        ruta
    ).convert(
        "RGB"
    )


    # ========================================================
    # ENFERMEDADES
    # ========================================================

    imagen_enfermedad = (
        imagen.resize(
            (160, 160),
            Image.Resampling.BILINEAR
        )
    )


    entrada_enfermedad = np.asarray(

        imagen_enfermedad,

        dtype=np.uint8

    )[None, ...]


    probabilidades_enfermedad = (

        modelo_enfermedades.predict(

            entrada_enfermedad,

            verbose=0

        )[0]
    )


    indice_enfermedad = int(

        np.argmax(
            probabilidades_enfermedad
        )
    )


    enfermedad = (

        CLASES_ENFERMEDADES[
            indice_enfermedad
        ]
    )


    prob_enfermedad = float(

        probabilidades_enfermedad[
            indice_enfermedad
        ]
    )


    info_enfermedad = (

        INFO_ENFERMEDADES[
            enfermedad
        ]
    )


    # ========================================================
    # RANKING ENFERMEDADES
    # ========================================================

    ranking_enfermedades = []


    for nombre, probabilidad in zip(

        CLASES_ENFERMEDADES,

        probabilidades_enfermedad
    ):

        ranking_enfermedades.append({

            "nombre":
                nombre,

            "probabilidad":
                float(
                    probabilidad
                )
        })


    ranking_enfermedades.sort(

        key=lambda item:
            item[
                "probabilidad"
            ],

        reverse=True
    )


    # ========================================================
    # ESTRÉS HÍDRICO
    # ========================================================

    imagen_agua = (
        imagen.resize(
            (48, 48),
            Image.Resampling.BILINEAR
        )
    )


    entrada_agua = np.asarray(

        imagen_agua,

        dtype=np.float32

    ) / 255.0


    entrada_agua = (
        entrada_agua[
            None,
            ...
        ]
    )


    probabilidades_agua = (

        modelo_agua.predict(

            entrada_agua,

            verbose=0

        )[0]
    )


    indice_agua = int(

        np.argmax(
            probabilidades_agua
        )
    )


    agua = (

        CLASES_AGUA[
            indice_agua
        ]
    )


    prob_agua = float(

        probabilidades_agua[
            indice_agua
        ]
    )


    # ========================================================
    # PLAGAS
    # ========================================================

    imagen_plaga = (
        imagen.resize(
            (160, 160),
            Image.Resampling.BILINEAR
        )
    )


    entrada_plaga = np.asarray(

        imagen_plaga,

        dtype=np.float32

    )[None, ...]


    probabilidades_plaga = (

        modelo_plagas.predict(

            entrada_plaga,

            verbose=0

        )[0]
    )


    prob_no_faw = float(

        probabilidades_plaga[0]
    )


    prob_faw = float(

        probabilidades_plaga[1]
    )


    # ========================================================
    # DECIDIR FALL ARMYWORM
    # ========================================================

    if prob_faw >= UMBRAL_PLAGA:

        plaga = (
            "Fall Armyworm "
            "(Gusano cogollero)"
        )

        info_plaga = (
            INFO_PLAGA_POSITIVA
        )


    else:

        plaga = (
            "Sin evidencia suficiente "
            "de Fall Armyworm"
        )

        info_plaga = (
            INFO_PLAGA_NEGATIVA
        )


    confianza_plaga = nivel_confianza(

        max(
            prob_faw,
            prob_no_faw
        )
    )


    # ========================================================
    # ADVERTENCIA HÍDRICA
    # ========================================================

    if (

        enfermedad
        != "Healthy (Sana)"

        and

        prob_enfermedad
        >= 0.70

    ):

        advertencia_agua = (

            "La imagen presenta una probabilidad elevada "
            "de enfermedad. Las lesiones pueden modificar "
            "el color y la textura de la hoja y afectar "
            "la predicción hídrica."
        )


    else:

        advertencia_agua = (

            "El resultado hídrico es experimental "
            "y debe interpretarse con precaución."
        )


    # ========================================================
    # DEVOLVER RESULTADO
    # ========================================================

    return {

        # Enfermedad

        "enfermedad":
            enfermedad,

        "nombre_enfermedad":
            info_enfermedad[
                "titulo"
            ],

        "prob_enfermedad":
            prob_enfermedad,

        "confianza_enfermedad":
            nivel_confianza(
                prob_enfermedad
            ),

        "descripcion_enfermedad":
            info_enfermedad[
                "descripcion"
            ],

        "sintomas_enfermedad":
            info_enfermedad[
                "sintomas"
            ],

        "recomendacion_enfermedad":
            info_enfermedad[
                "recomendacion"
            ],

        "ranking_enfermedades":
            ranking_enfermedades,


        # Agua

        "agua":
            agua,

        "prob_agua":
            prob_agua,

        "confianza_agua":
            nivel_confianza(
                prob_agua
            ),

        "advertencia_agua":
            advertencia_agua,


        # Plagas

        "plaga":
            plaga,

        "prob_plaga":
            prob_faw,

        "prob_no_plaga":
            prob_no_faw,

        "confianza_plaga":
            confianza_plaga,

        "titulo_plaga":
            info_plaga[
                "titulo"
            ],

        "descripcion_plaga":
            info_plaga[
                "descripcion"
            ],

        "sintomas_plaga":
            info_plaga[
                "sintomas"
            ],

        "recomendacion_plaga":
            info_plaga[
                "recomendacion"
            ]
    }