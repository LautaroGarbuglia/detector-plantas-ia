"""
CONTROLLER - Feedback

Este controlador maneja:

- Resultado correcto.
- Resultado incorrecto.
- Formulario de corrección.

La escritura en SQLite se delega al Model.
"""

from flask import (
    Blueprint,
    request,
    redirect,
    url_for,
    render_template
)


from app.models.analisis_model import (

    obtener_analisis,

    guardar_feedback_correcto,

    guardar_correccion
)


# ============================================================
# BLUEPRINT
# ============================================================

feedback_bp = Blueprint(
    "feedback",
    __name__
)


# ============================================================
# FEEDBACK
# ============================================================

@feedback_bp.route(
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
    # CORRECTO
    # ========================================================

    if valor == "correcto":

        guardar_feedback_correcto(
            analisis_id
        )


        return redirect(

            url_for(

                "analisis.inicio",

                mensaje=(
                    "✅ Gracias. "
                    "La valoración fue "
                    "guardada correctamente."
                )
            )
        )


    # ========================================================
    # INCORRECTO
    # ========================================================

    if valor == "incorrecto":

        return redirect(

            url_for(

                "feedback.corregir",

                analisis_id=
                    analisis_id
            )
        )


    return redirect(

        url_for(
            "analisis.inicio"
        )
    )


# ============================================================
# CORREGIR
# ============================================================

@feedback_bp.route(
    "/corregir/<analisis_id>",
    methods=[
        "GET",
        "POST"
    ]
)
def corregir(
    analisis_id
):

    # ========================================================
    # MODEL
    # ========================================================

    analisis = obtener_analisis(
        analisis_id
    )


    if analisis is None:

        return (
            "Análisis no encontrado.",
            404
        )


    # ========================================================
    # GUARDAR CORRECCIÓN
    # ========================================================

    if request.method == "POST":

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
        # VALIDAR
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
                    "un resultado incorrecto."
                )
            )


        # ====================================================
        # RESPUESTAS CORRECTAS
        # ====================================================

        enfermedad_correcta = (

            request.form.get(
                "enfermedad_correcta"
            )

            if error_enfermedad

            else None
        )


        agua_correcta = (

            request.form.get(
                "agua_correcta"
            )

            if error_agua

            else None
        )


        plaga_correcta = (

            request.form.get(
                "plaga_correcta"
            )

            if error_plaga

            else None
        )


        # ====================================================
        # MODEL
        # ====================================================

        guardar_correccion(

            analisis_id,

            error_enfermedad,
            enfermedad_correcta,

            error_agua,
            agua_correcta,

            error_plaga,
            plaga_correcta
        )


        return redirect(

            url_for(

                "analisis.inicio",

                mensaje=(
                    "✅ Corrección guardada. "
                    "Podrá utilizarse para "
                    "futuros reentrenamientos."
                )
            )
        )


    # ========================================================
    # MOSTRAR VIEW
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