"""
CONTROLLER - Análisis

Responsabilidades:

- Mostrar la página principal.
- Recibir imágenes.
- Validar archivos.
- Pedirle al servicio de IA que analice.
- Pedirle al Model que guarde los resultados.
- Enviar información a la View.

El controlador NO contiene SQL
y tampoco implementa redes neuronales.
"""

from flask import (
    Blueprint,
    render_template,
    request,
    url_for,
    current_app
)

from pathlib import Path

from PIL import (
    Image,
    UnidentifiedImageError
)

import uuid


from app.services.ia_service import (
    analizar_imagen
)

from app.models.analisis_model import (
    crear_analisis
)


# ============================================================
# BLUEPRINT
# ============================================================

analisis_bp = Blueprint(
    "analisis",
    __name__
)


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@analisis_bp.route(
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
    # SUBIR IMAGEN
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
                not in extensiones_permitidas
            ):

                return render_template(

                    "index.html",

                    mensaje=(
                        "⚠️ Formato de imagen "
                        "no permitido."
                    )
                )


            # =================================================
            # NOMBRE ÚNICO
            # =================================================

            nombre = (

                uuid.uuid4().hex
                + extension
            )


            ruta = (

                current_app.config[
                    "UPLOAD_DIR"
                ]

                / nombre
            )


            archivo.save(
                ruta
            )


            # =================================================
            # VALIDAR IMAGEN
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
            # SERVICE
            # =================================================

            resultado = analizar_imagen(
                ruta
            )


            # =================================================
            # MODEL
            # =================================================

            analisis_id = crear_analisis(

                nombre,

                resultado
            )


            # =================================================
            # URL DE LA IMAGEN
            # =================================================

            imagen_url = url_for(

                "static",

                filename=(
                    f"uploads/{nombre}"
                )
            )


    # ========================================================
    # VIEW
    # ========================================================

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