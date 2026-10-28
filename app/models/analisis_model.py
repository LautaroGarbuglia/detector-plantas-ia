"""
MODEL - AnalisisModel

Este archivo contiene toda la comunicación
con la base de datos SQLite.

Responsabilidades:

- Crear la tabla.
- Guardar análisis.
- Buscar análisis.
- Guardar feedback.
- Guardar correcciones.

Los controladores NO deberían escribir SQL directamente.
"""

import sqlite3
import uuid

from datetime import datetime
from flask import current_app


# ============================================================
# OBTENER CONEXIÓN
# ============================================================

def obtener_conexion():

    conexion = sqlite3.connect(
        current_app.config[
            "DB_PATH"
        ]
    )

    conexion.row_factory = (
        sqlite3.Row
    )

    return conexion


# ============================================================
# INICIALIZAR BASE
# ============================================================

def inicializar_db():

    conexion = obtener_conexion()

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
    # VER COLUMNAS ACTUALES
    # ========================================================

    cursor.execute(
        "PRAGMA table_info(analisis)"
    )

    columnas = {

        fila["name"]

        for fila
        in cursor.fetchall()
    }


    # ========================================================
    # COLUMNAS DE FEEDBACK
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
    # AGREGAR COLUMNAS FALTANTES
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
                f"Columna agregada: {nombre}"
            )


    conexion.commit()

    conexion.close()


# ============================================================
# CREAR ANÁLISIS
# ============================================================

def crear_analisis(
    imagen,
    resultado
):

    analisis_id = (
        uuid.uuid4().hex
    )


    conexion = obtener_conexion()

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

            imagen,

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


    return analisis_id


# ============================================================
# OBTENER UN ANÁLISIS
# ============================================================

def obtener_analisis(
    analisis_id
):

    conexion = obtener_conexion()

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


    analisis = cursor.fetchone()


    conexion.close()


    return analisis


# ============================================================
# FEEDBACK CORRECTO
# ============================================================

def guardar_feedback_correcto(
    analisis_id
):

    conexion = obtener_conexion()

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


# ============================================================
# GUARDAR CORRECCIÓN
# ============================================================

def guardar_correccion(

    analisis_id,

    error_enfermedad,
    enfermedad_correcta,

    error_agua,
    agua_correcta,

    error_plaga,
    plaga_correcta
):

    conexion = obtener_conexion()

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