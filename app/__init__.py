"""
Configuración principal de la aplicación Flask.

Este archivo se encarga de:

- Crear Flask.
- Configurar templates.
- Configurar archivos estáticos.
- Configurar uploads.
- Configurar SQLite.
- Inicializar la base de datos.
- Registrar los controladores.
"""

from pathlib import Path

from flask import Flask

from app.models.analisis_model import (
    inicializar_db
)


def create_app():

    # ========================================================
    # RUTA DE LA CARPETA app/
    # ========================================================

    app_dir = Path(
        __file__
    ).resolve().parent


    # ========================================================
    # RUTA RAÍZ DEL PROYECTO
    # ========================================================

    project_root = (
        app_dir.parent
    )


    # ========================================================
    # CREAR FLASK
    # ========================================================
    #
    # Usamos rutas ABSOLUTAS para evitar cualquier problema
    # encontrando templates o archivos static.
    #

    flask_app = Flask(

        __name__,

        template_folder=str(
            app_dir / "templates"
        ),

        static_folder=str(
            app_dir / "static"
        )
    )


    # ========================================================
    # BASE DE DATOS
    # ========================================================

    flask_app.config[
        "DB_PATH"
    ] = (

        project_root
        / "feedback.db"
    )


    # ========================================================
    # CARPETA DE IMÁGENES
    # ========================================================

    flask_app.config[
        "UPLOAD_DIR"
    ] = (

        app_dir
        / "static"
        / "uploads"
    )


    # Crear uploads si no existe.

    flask_app.config[
        "UPLOAD_DIR"
    ].mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # MOSTRAR RUTAS EN CONSOLA
    # ========================================================
    #
    # Esto nos sirve para comprobar exactamente dónde
    # está buscando Flask los archivos.
    #

    print()
    print("=" * 60)

    print(
        "Carpeta APP:",
        app_dir
    )

    print(
        "Templates:",
        app_dir / "templates"
    )

    print(
        "Static:",
        app_dir / "static"
    )

    print(
        "Uploads:",
        flask_app.config["UPLOAD_DIR"]
    )

    print(
        "Base de datos:",
        flask_app.config["DB_PATH"]
    )

    print("=" * 60)
    print()


    # ========================================================
    # INICIALIZAR SQLITE
    # ========================================================

    with flask_app.app_context():

        inicializar_db()


    # ========================================================
    # IMPORTAR CONTROLADORES
    # ========================================================

    from app.controllers.analisis_controller import (
        analisis_bp
    )

    from app.controllers.feedback_controller import (
        feedback_bp
    )


    # ========================================================
    # REGISTRAR CONTROLADORES
    # ========================================================

    flask_app.register_blueprint(
        analisis_bp
    )

    flask_app.register_blueprint(
        feedback_bp
    )


    # ========================================================
    # MOSTRAR RUTAS FLASK
    # ========================================================

    print(
        "Rutas registradas:"
    )

    print(
        flask_app.url_map
    )

    print()


    return flask_app