from flask import (
    Flask,
    render_template,
    request,
    url_for,
    redirect
)

from pathlib import Path
from PIL import Image, UnidentifiedImageError

import numpy as np
import sqlite3
import uuid

from datetime import datetime


# ============================================================
# IMPORTAR LOS MODELOS
# ============================================================
#
# Cada archivo .py contiene:
#
# - La arquitectura del modelo.
# - Información sobre sus clases.
# - Una función cargar_modelo().
#
# Los pesos aprendidos durante el entrenamiento están
# almacenados por separado en archivos .weights.h5.
#

from modelos.enfermedades import (
    cargar_modelo as cargar_modelo_enfermedades,
    CLASES as CLASES_ENFERMEDADES
)

from modelos.agua import (
    cargar_modelo as cargar_modelo_agua,
    CLASES as CLASES_AGUA
)

from modelos.plagas import (
    cargar_modelo as cargar_modelo_plagas,
    UMBRAL as UMBRAL_PLAGA
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

app = Flask(__name__)


# Carpeta principal del proyecto.
BASE_DIR = Path(__file__).resolve().parent


# Carpeta donde se guardan temporalmente
# las imágenes que suben los usuarios.
UPLOAD_DIR = (
    BASE_DIR
    / "static"
    / "uploads"
)


# Base de datos donde almacenamos:
#
# - Predicciones.
# - Probabilidades.
# - Feedback.
# - Correcciones de usuarios.
#
DB_PATH = (
    BASE_DIR
    / "feedback.db"
)


# Crear la carpeta uploads si todavía no existe.
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARGAR LOS TRES MODELOS
# ============================================================
#
# IMPORTANTE:
#
# Los modelos se cargan una sola vez cuando inicia Flask.
#
# No se vuelven a cargar cada vez que alguien analiza
# una fotografía.
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
    "✅ Modelos cargados correctamente."
)


# ============================================================
# INFORMACIÓN SOBRE LAS ENFERMEDADES
# ============================================================
#
# Esta información NO modifica el modelo.
#
# Solamente se utiliza para explicar los resultados
# de una forma más comprensible para el usuario.
#

INFO_ENFERMEDADES = {

    "Blight (Tizón)": {

        "titulo":
            "Tizón foliar",

        "descripcion": (
            "El modelo encontró características visuales "
            "similares a las hojas de maíz clasificadas "
            "como tizón durante su entrenamiento."
        ),

        "sintomas": (
            "Puede presentar lesiones alargadas, zonas "
            "marrones o tejido seco que se extiende "
            "a lo largo de la hoja."
        ),

        "recomendacion": (
            "Revisar otras hojas de la misma planta y "
            "observar si aparecen lesiones alargadas o "
            "nuevas áreas secas. Si el problema se "
            "extiende, conviene realizar una evaluación "
            "agronómica."
        )
    },


    "Common Rust (Roya común)": {

        "titulo":
            "Roya común",

        "descripcion": (
            "La imagen presenta características similares "
            "a las hojas utilizadas para entrenar la clase "
            "Roya común."
        ),

        "sintomas": (
            "Suelen observarse pequeñas pústulas o manchas "
            "marrones, rojizas o anaranjadas distribuidas "
            "sobre la superficie de la hoja."
        ),

        "recomendacion": (
            "Revisar ambas caras de otras hojas de la "
            "planta y comprobar si existen pústulas "
            "o manchas similares."
        )
    },


    "Gray Leaf Spot (Mancha gris)": {

        "titulo":
            "Mancha gris",

        "descripcion": (
            "El modelo encontró características compatibles "
            "con las imágenes de mancha gris utilizadas "
            "durante su entrenamiento."
        ),

        "sintomas": (
            "Pueden observarse lesiones alargadas o "
            "rectangulares de color grisáceo o marrón, "
            "frecuentemente ubicadas entre las "
            "nervaduras de la hoja."
        ),

        "recomendacion": (
            "Observar otras hojas y verificar si aparecen "
            "manchas alargadas o rectangulares semejantes."
        )
    },


    "Healthy (Sana)": {

        "titulo":
            "Hoja aparentemente sana",

        "descripcion": (
            "El modelo encontró mayor similitud con las "
            "hojas sanas utilizadas durante su "
            "entrenamiento."
        ),

        "sintomas": (
            "No se detectaron características dominantes "
            "correspondientes a las tres enfermedades "
            "que reconoce este modelo."
        ),

        "recomendacion": (
            "Continuar observando la planta. Este resultado "
            "no descarta problemas, deficiencias o "
            "enfermedades que el modelo todavía no fue "
            "entrenado para reconocer."
        )
    }
}


# ============================================================
# INFORMACIÓN SOBRE FALL ARMYWORM
# ============================================================


INFO_PLAGA_POSITIVA = {

    "titulo":
        "Posible presencia de gusano cogollero",

    "descripcion": (
        "El modelo encontró características visuales "
        "compatibles con las imágenes de Fall Armyworm "
        "utilizadas durante su entrenamiento."
    ),

    "sintomas": (
        "Conviene revisar si existen perforaciones, "
        "tejido raspado, bordes dañados, orificios "
        "irregulares o daños concentrados cerca "
        "del cogollo."
    ),

    "recomendacion": (
        "Inspeccionar manualmente la planta y, "
        "especialmente, el cogollo. La predicción visual "
        "debe utilizarse como una señal para realizar "
        "una revisión más detallada."
    )
}


INFO_PLAGA_NEGATIVA = {

    "titulo":
        "Sin evidencia suficiente de gusano cogollero",

    "descripcion": (
        "La probabilidad calculada por el modelo se "
        "encuentra por debajo del umbral utilizado para "
        "clasificar la imagen como Fall Armyworm."
    ),

    "sintomas": (
        "El modelo no encontró suficientes características "
        "similares a los ejemplos de gusano cogollero "
        "incluidos en su entrenamiento."
    ),

    "recomendacion": (
        "Esto no descarta otras plagas ni garantiza que "
        "la planta esté completamente sana. Si existen "
        "daños visibles, conviene inspeccionar la planta "
        "manualmente."
    )
}


# ============================================================
# NIVEL ORIENTATIVO DE CONFIANZA
# ============================================================
#
# Esto NO representa una certeza diagnóstica.
#
# Simplemente transforma la probabilidad de salida
# del modelo en una etiqueta más fácil de interpretar.
#

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
# CREAR / ACTUALIZAR BASE DE DATOS
# ============================================================

def inicializar_db():

    conexion = sqlite3.connect(
        DB_PATH
    )

    cursor = conexion.cursor()


    # ========================================================
    # TABLA PRINCIPAL
    # ========================================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analisis (

            id TEXT PRIMARY KEY,

            imagen TEXT NOT NULL,

            enfermedad TEXT,
            prob_enfermedad REAL,

            agua TEXT,
            prob_agua REAL,

            plaga TEXT,
            prob_plaga REAL,

            feedback TEXT,

            fecha TEXT
        )
        """
    )


    # ========================================================
    # COMPROBAR COLUMNAS EXISTENTES
    # ========================================================

    cursor.execute(
        "PRAGMA table_info(analisis)"
    )


    columnas = {

        fila[1]

        for fila in cursor.fetchall()
    }


    # ========================================================
    # COLUMNAS NECESARIAS PARA EL FEEDBACK
    # ========================================================

    nuevas_columnas = {

        "error_enfermedad":
            "INTEGER DEFAULT 0",

        "enfermedad_correcta":
            "TEXT",

        "error_agua":
            "INTEGER DEFAULT 0",

        "agua_correcta":
            "TEXT",

        "error_plaga":
            "INTEGER DEFAULT 0",

        "plaga_correcta":
            "TEXT",

        "fecha_feedback":
            "TEXT"
    }


    # ========================================================
    # AGREGAR LAS QUE FALTEN
    # ========================================================

    for nombre, tipo in (
        nuevas_columnas.items()
    ):

        if nombre not in columnas:

            cursor.execute(
                f"""
                ALTER TABLE analisis
                ADD COLUMN {nombre} {tipo}
                """
            )


            print(
                "Columna agregada:",
                nombre
            )


    conexion.commit()

    conexion.close()


# Crear / actualizar DB al arrancar.
inicializar_db()


# ============================================================
# ANALIZAR UNA IMAGEN
# ============================================================

def analizar_imagen(
    ruta
):

    # Abrimos la imagen y la convertimos
    # siempre a RGB.
    imagen = Image.open(
        ruta
    ).convert(
        "RGB"
    )


    # ========================================================
    # MODELO 1:
    # ENFERMEDADES
    # ========================================================
    #
    # Entrada esperada:
    #
    # 160 x 160 x 3
    #
    # El modelo posee su propio preprocesamiento interno.
    # Por eso NO dividimos los píxeles por 255.
    # ========================================================

    img_enfermedad = imagen.resize(

        (160, 160),

        Image.Resampling.BILINEAR
    )


    entrada_enfermedad = np.asarray(

        img_enfermedad,

        dtype=np.uint8
    )


    # Tensor:
    #
    # (160,160,3)
    #
    # pasa a:
    #
    # (1,160,160,3)
    #
    entrada_enfermedad = np.expand_dims(

        entrada_enfermedad,

        axis=0
    )


    probabilidades_enfermedad = (

        modelo_enfermedades.predict(

            entrada_enfermedad,

            verbose=0

        )[0]
    )


    # Elegimos la clase que tenga
    # mayor probabilidad.
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
    # RANKING DE TODAS LAS ENFERMEDADES
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


    # Ordenar de mayor a menor.
    ranking_enfermedades.sort(

        key=lambda elemento:
            elemento[
                "probabilidad"
            ],

        reverse=True
    )


    # ========================================================
    # MODELO 2:
    # ESTRÉS HÍDRICO
    # ========================================================
    #
    # Entrada esperada:
    #
    # 48 x 48 x 3
    #
    # Este modelo fue entrenado utilizando
    # valores normalizados entre 0 y 1.
    #
    # Por eso dividimos por 255.
    # ========================================================

    img_agua = imagen.resize(

        (48, 48),

        Image.Resampling.BILINEAR
    )


    entrada_agua = np.asarray(

        img_agua,

        dtype=np.float32
    )


    entrada_agua = (

        entrada_agua
        / 255.0
    )


    entrada_agua = np.expand_dims(

        entrada_agua,

        axis=0
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
    # MODELO 3:
    # FALL ARMYWORM / GUSANO COGOLLERO
    # ========================================================
    #
    # Entrada esperada:
    #
    # 160 x 160 x 3
    #
    # El modelo posee un Rescaling interno.
    #
    # Por eso NO dividimos por 255.
    # ========================================================

    img_plaga = imagen.resize(

        (160, 160),

        Image.Resampling.BILINEAR
    )


    entrada_plaga = np.asarray(

        img_plaga,

        dtype=np.float32
    )


    entrada_plaga = np.expand_dims(

        entrada_plaga,

        axis=0
    )


    probabilidades_plaga = (

        modelo_plagas.predict(

            entrada_plaga,

            verbose=0

        )[0]
    )


    # Clase 0:
    # No Fall Armyworm
    prob_no_faw = float(

        probabilidades_plaga[0]
    )


    # Clase 1:
    # Fall Armyworm
    prob_faw = float(

        probabilidades_plaga[1]
    )


    # ========================================================
    # DECISIÓN DE PLAGA
    # ========================================================
    #
    # No usamos simplemente 50%.
    #
    # Durante la validación del modelo se fijó
    # el umbral en 34%.
    #
    # Si:
    #
    # prob_faw >= 0.34
    #
    # clasificamos como Fall Armyworm.
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


    # ========================================================
    # NIVEL ORIENTATIVO DE PLAGA
    # ========================================================

    confianza_plaga = nivel_confianza(

        max(

            prob_faw,

            prob_no_faw
        )
    )


    # ========================================================
    # ADVERTENCIA PARA EL MODELO DE AGUA
    # ========================================================
    #
    # Si existe una enfermedad con alta probabilidad,
    # las lesiones pueden alterar color y textura.
    #
    # Eso puede afectar la interpretación del
    # modelo hídrico.
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
            "la predicción del modelo hídrico."
        )


    else:

        advertencia_agua = (

            "El resultado hídrico es experimental y "
            "debe interpretarse con precaución."
        )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    return {

        # ====================================================
        # ENFERMEDAD
        # ====================================================

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


        # ====================================================
        # AGUA
        # ====================================================

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


        # ====================================================
        # PLAGA
        # ====================================================

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


# ============================================================
# BUSCAR ANÁLISIS EN SQLITE
# ============================================================

def obtener_analisis(
    analisis_id
):

    conexion = sqlite3.connect(
        DB_PATH
    )


    # Esto permite acceder a las columnas así:
    #
    # analisis["imagen"]
    #
    # en vez de:
    #
    # analisis[1]
    #
    conexion.row_factory = (
        sqlite3.Row
    )


    cursor = conexion.cursor()


    cursor.execute(
        """
        SELECT *
        FROM analisis
        WHERE id = ?
        """,

        (
            analisis_id,
        )
    )


    fila = cursor.fetchone()


    conexion.close()


    return fila


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def inicio():

    imagen_url = None

    resultado = None

    analisis_id = None


    mensaje = request.args.get(
        "mensaje"
    )


    # ========================================================
    # SI SE SUBE UNA IMAGEN
    # ========================================================

    if request.method == "POST":

        archivo = request.files.get(
            "imagen"
        )


        if (
            archivo
            and archivo.filename
        ):

            # =================================================
            # EXTENSIÓN
            # =================================================

            extension = Path(

                archivo.filename

            ).suffix.lower()


            extensiones_permitidas = {

                ".jpg",

                ".jpeg",

                ".png",

                ".webp"
            }


            if (
                extension
                not in
                extensiones_permitidas
            ):

                return render_template(

                    "index.html",

                    mensaje=(

                        "⚠️ Formato de imagen "
                        "no permitido."
                    )
                )


            # =================================================
            # GENERAR NOMBRE ÚNICO
            # =================================================

            nombre = (

                uuid.uuid4().hex
                + extension
            )


            ruta = (

                UPLOAD_DIR
                / nombre
            )


            # Guardar imagen.
            archivo.save(
                ruta
            )


            # =================================================
            # COMPROBAR QUE REALMENTE SEA UNA IMAGEN
            # =================================================

            try:

                with Image.open(
                    ruta
                ) as prueba:

                    prueba.verify()


            except (
                UnidentifiedImageError,
                OSError
            ):

                ruta.unlink(
                    missing_ok=True
                )


                return render_template(

                    "index.html",

                    mensaje=(

                        "⚠️ El archivo seleccionado "
                        "no es una imagen válida."
                    )
                )


            # =================================================
            # ANALIZAR CON LOS TRES MODELOS
            # =================================================

            resultado = analizar_imagen(
                ruta
            )


            # =================================================
            # ID ÚNICO DEL ANÁLISIS
            # =================================================

            analisis_id = (
                uuid.uuid4().hex
            )


            # =================================================
            # URL DE LA FOTO
            # =================================================

            imagen_url = url_for(

                "static",

                filename=(

                    f"uploads/{nombre}"
                )
            )


            # =================================================
            # GUARDAR RESULTADOS EN SQLITE
            # =================================================

            conexion = sqlite3.connect(
                DB_PATH
            )


            cursor = conexion.cursor()


            cursor.execute(
                """
                INSERT INTO analisis (

                    id,

                    imagen,

                    enfermedad,

                    prob_enfermedad,

                    agua,

                    prob_agua,

                    plaga,

                    prob_plaga,

                    feedback,

                    fecha

                )

                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,

                (

                    analisis_id,

                    nombre,

                    resultado[
                        "enfermedad"
                    ],

                    resultado[
                        "prob_enfermedad"
                    ],

                    resultado[
                        "agua"
                    ],

                    resultado[
                        "prob_agua"
                    ],

                    resultado[
                        "plaga"
                    ],

                    resultado[
                        "prob_plaga"
                    ],

                    None,

                    datetime.now().isoformat()
                )
            )


            conexion.commit()

            conexion.close()


    return render_template(

        "index.html",

        imagen_url=
            imagen_url,

        resultado=
            resultado,

        analisis_id=
            analisis_id,

        mensaje=
            mensaje
    )


# ============================================================
# FEEDBACK
# ============================================================

@app.route(
    "/feedback",
    methods=["POST"]
)
def feedback():

    analisis_id = request.form.get(
        "analisis_id"
    )


    valor = request.form.get(
        "valor"
    )


    # ========================================================
    # RESULTADO CORRECTO
    # ========================================================

    if valor == "correcto":

        conexion = sqlite3.connect(
            DB_PATH
        )


        cursor = conexion.cursor()


        cursor.execute(
            """
            UPDATE analisis

            SET

                feedback = ?,

                fecha_feedback = ?

            WHERE id = ?
            """,

            (

                "correcto",

                datetime.now().isoformat(),

                analisis_id
            )
        )


        conexion.commit()

        conexion.close()


        return redirect(

            url_for(

                "inicio",

                mensaje=(

                    "✅ Gracias. "
                    "La valoración fue "
                    "guardada correctamente."
                )
            )
        )


    # ========================================================
    # RESULTADO INCORRECTO
    # ========================================================

    if valor == "incorrecto":

        return redirect(

            url_for(

                "corregir",

                analisis_id=
                    analisis_id
            )
        )


    return redirect(
        url_for(
            "inicio"
        )
    )


# ============================================================
# CORREGIR RESULTADO
# ============================================================

@app.route(
    "/corregir/<analisis_id>",
    methods=[
        "GET",
        "POST"
    ]
)
def corregir(
    analisis_id
):

    # Buscar el análisis original.
    analisis = obtener_analisis(
        analisis_id
    )


    if analisis is None:

        return (

            "Análisis no encontrado.",

            404
        )


    # ========================================================
    # SI EL USUARIO ENVÍA UNA CORRECCIÓN
    # ========================================================

    if request.method == "POST":

        # ====================================================
        # ¿QUÉ RESULTADOS ESTABAN MAL?
        # ====================================================


        error_enfermedad = (

            1

            if request.form.get(
                "error_enfermedad"
            )

            else 0
        )


        error_agua = (

            1

            if request.form.get(
                "error_agua"
            )

            else 0
        )


        error_plaga = (

            1

            if request.form.get(
                "error_plaga"
            )

            else 0
        )


        # ====================================================
        # AL MENOS UNA OPCIÓN DEBE ESTAR MARCADA
        # ====================================================

        if not (

            error_enfermedad

            or error_agua

            or error_plaga
        ):

            imagen_url = url_for(

                "static",

                filename=(

                    "uploads/"

                    + analisis[
                        "imagen"
                    ]
                )
            )


            return render_template(

                "corregir.html",

                analisis=
                    analisis,

                imagen_url=
                    imagen_url,

                error=(

                    "Seleccioná al menos "
                    "un resultado que "
                    "consideres incorrecto."
                )
            )


        # ====================================================
        # RESPUESTA CORRECTA DE ENFERMEDAD
        # ====================================================

        enfermedad_correcta = (

            request.form.get(
                "enfermedad_correcta"
            )

            if error_enfermedad

            else None
        )


        # ====================================================
        # RESPUESTA CORRECTA DE AGUA
        # ====================================================

        agua_correcta = (

            request.form.get(
                "agua_correcta"
            )

            if error_agua

            else None
        )


        # ====================================================
        # RESPUESTA CORRECTA DE PLAGA
        # ====================================================

        plaga_correcta = (

            request.form.get(
                "plaga_correcta"
            )

            if error_plaga

            else None
        )


        # ====================================================
        # GUARDAR CORRECCIÓN
        # ====================================================

        conexion = sqlite3.connect(
            DB_PATH
        )


        cursor = conexion.cursor()


        cursor.execute(
            """
            UPDATE analisis

            SET

                feedback = ?,

                error_enfermedad = ?,

                enfermedad_correcta = ?,

                error_agua = ?,

                agua_correcta = ?,

                error_plaga = ?,

                plaga_correcta = ?,

                fecha_feedback = ?

            WHERE id = ?
            """,

            (

                "incorrecto",

                error_enfermedad,

                enfermedad_correcta,

                error_agua,

                agua_correcta,

                error_plaga,

                plaga_correcta,

                datetime.now().isoformat(),

                analisis_id
            )
        )


        conexion.commit()

        conexion.close()


        return redirect(

            url_for(

                "inicio",

                mensaje=(

                    "✅ Corrección guardada. "
                    "Podrá utilizarse para "
                    "futuros reentrenamientos."
                )
            )
        )


    # ========================================================
    # MOSTRAR FORMULARIO DE CORRECCIÓN
    # ========================================================

    imagen_url = url_for(

        "static",

        filename=(

            "uploads/"

            + analisis[
                "imagen"
            ]
        )
    )


    return render_template(

        "corregir.html",

        analisis=
            analisis,

        imagen_url=
            imagen_url
    )


# ============================================================
# INICIAR APLICACIÓN
# ============================================================

if __name__ == "__main__":

    app.run(

        host=
            "127.0.0.1",

        port=
            5000,

        debug=
            True,

        # Evitamos que Flask reinicie el proceso
        # y cargue los tres modelos dos veces.
        use_reloader=
            False
    )