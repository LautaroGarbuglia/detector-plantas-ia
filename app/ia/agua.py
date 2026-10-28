"""
MODELO: AGUA

Este archivo representa la arquitectura del modelo entrenado.

NO entrena el modelo.

Los pesos aprendidos se encuentran en:

    agua.weights.h5

Función:
    Clasificación experimental de patrones hídricos.

Entrada:
    (None, 48, 48, 3)

Salida:
    (None, 3)

Parámetros:
    406,499

Preprocesamiento:
    Imagen RGB de 48x48. Antes de ingresar al modelo se divide por 255.
"""

from pathlib import Path
import tensorflow as tf


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

DESCRIPCION = 'Clasificación experimental de patrones hídricos.'

CLASES = ['Clase 0', 'Clase 1', 'Clase 2']

PREPROCESAMIENTO = 'Imagen RGB de 48x48. Antes de ingresar al modelo se divide por 255.'

INPUT_SHAPE = (None, 48, 48, 3)

OUTPUT_SHAPE = (None, 3)

PARAMETROS = 406499


# ============================================================
# CAPAS PRINCIPALES DEL MODELO
# ============================================================
#
# La siguiente lista permite entender visualmente las capas
# principales sin tener que interpretar el archivo de pesos.
#
# 00. input_layer | InputLayer | Salida: (None, 48, 48, 3)
# 01. data_augmentation | Sequential | Salida: (None, 48, 48, 3)
# 02. conv2d | Conv2D | Salida: (None, 48, 48, 32)
# 03. batch_normalization | BatchNormalization | Salida: (None, 48, 48, 32)
# 04. activation | Activation | Salida: (None, 48, 48, 32)
# 05. max_pooling2d | MaxPooling2D | Salida: (None, 24, 24, 32)
# 06. conv2d_1 | Conv2D | Salida: (None, 24, 24, 64)
# 07. batch_normalization_1 | BatchNormalization | Salida: (None, 24, 24, 64)
# 08. activation_1 | Activation | Salida: (None, 24, 24, 64)
# 09. max_pooling2d_1 | MaxPooling2D | Salida: (None, 12, 12, 64)
# 10. conv2d_2 | Conv2D | Salida: (None, 12, 12, 128)
# 11. batch_normalization_2 | BatchNormalization | Salida: (None, 12, 12, 128)
# 12. activation_2 | Activation | Salida: (None, 12, 12, 128)
# 13. max_pooling2d_2 | MaxPooling2D | Salida: (None, 6, 6, 128)
# 14. conv2d_3 | Conv2D | Salida: (None, 6, 6, 256)
# 15. batch_normalization_3 | BatchNormalization | Salida: (None, 6, 6, 256)
# 16. activation_3 | Activation | Salida: (None, 6, 6, 256)
# 17. global_average_pooling2d | GlobalAveragePooling2D | Salida: (None, 256)
# 18. dropout | Dropout | Salida: (None, 256)
# 19. dense | Dense | Salida: (None, 64)
# 20. dropout_1 | Dropout | Salida: (None, 64)
# 21. dense_1 | Dense | Salida: (None, 3)


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

MODEL_JSON = '{"module": "keras.src.models.functional", "class_name": "Functional", "config": {"name": "modelo_agua_v2", "trainable": true, "layers": [{"module": "keras.layers", "class_name": "InputLayer", "config": {"batch_shape": [null, 48, 48, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer", "optional": false}, "registered_name": null, "name": "input_layer", "inbound_nodes": []}, {"module": "keras", "class_name": "Sequential", "config": {"name": "data_augmentation", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "layers": [{"module": "keras.layers", "class_name": "InputLayer", "config": {"batch_shape": [null, 48, 48, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer_1", "optional": false}, "registered_name": null}, {"module": "keras.layers", "class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "seed": null, "mode": "horizontal", "data_format": "channels_last"}, "registered_name": null, "build_config": {"input_shape": [null, 48, 48, 3]}}, {"module": "keras.layers", "class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "factor": [-0.05, 0.05], "data_format": "channels_last", "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}, "registered_name": null, "build_config": {"input_shape": [null, 48, 48, 3]}}, {"module": "keras.layers", "class_name": "RandomZoom", "config": {"name": "random_zoom", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "height_factor": 0.05, "width_factor": null, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null, "fill_value": 0.0, "data_format": "channels_last"}, "registered_name": null, "build_config": {"input_shape": [null, 48, 48, 3]}}], "build_input_shape": [null, 48, 48, 3]}, "registered_name": null, "build_config": {"input_shape": [null, 48, 48, 3]}, "compile_config": {}, "name": "data_augmentation", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 48, 48, 3], "dtype": "float32", "keras_history": ["input_layer", 0, 0]}}], "kwargs": {"mask": null}}]}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 48, 48, 3]}, "name": "conv2d", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 48, 48, 3], "dtype": "float32", "keras_history": ["data_augmentation", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 48, 48, 32]}, "name": "batch_normalization", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 48, 48, 32], "dtype": "float32", "keras_history": ["conv2d", 0, 0]}}], "kwargs": {"mask": null}}]}, {"module": "keras.layers", "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "activation": "relu"}, "registered_name": null, "name": "activation", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 48, 48, 32], "dtype": "float32", "keras_history": ["batch_normalization", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "registered_name": null, "name": "max_pooling2d", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 48, 48, 32], "dtype": "float32", "keras_history": ["activation", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 24, 24, 32]}, "name": "conv2d_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 24, 24, 32], "dtype": "float32", "keras_history": ["max_pooling2d", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 24, 24, 64]}, "name": "batch_normalization_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 24, 24, 64], "dtype": "float32", "keras_history": ["conv2d_1", 0, 0]}}], "kwargs": {"mask": null}}]}, {"module": "keras.layers", "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "activation": "relu"}, "registered_name": null, "name": "activation_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 24, 24, 64], "dtype": "float32", "keras_history": ["batch_normalization_1", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "registered_name": null, "name": "max_pooling2d_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 24, 24, 64], "dtype": "float32", "keras_history": ["activation_1", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 12, 12, 64]}, "name": "conv2d_2", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 12, 12, 64], "dtype": "float32", "keras_history": ["max_pooling2d_1", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 12, 12, 128]}, "name": "batch_normalization_2", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 12, 12, 128], "dtype": "float32", "keras_history": ["conv2d_2", 0, 0]}}], "kwargs": {"mask": null}}]}, {"module": "keras.layers", "class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "activation": "relu"}, "registered_name": null, "name": "activation_2", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 12, 12, 128], "dtype": "float32", "keras_history": ["batch_normalization_2", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "registered_name": null, "name": "max_pooling2d_2", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 12, 12, 128], "dtype": "float32", "keras_history": ["activation_2", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 6, 6, 128]}, "name": "conv2d_3", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 6, 6, 128], "dtype": "float32", "keras_history": ["max_pooling2d_2", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 6, 6, 256]}, "name": "batch_normalization_3", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 6, 6, 256], "dtype": "float32", "keras_history": ["conv2d_3", 0, 0]}}], "kwargs": {"mask": null}}]}, {"module": "keras.layers", "class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "activation": "relu"}, "registered_name": null, "name": "activation_3", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 6, 6, 256], "dtype": "float32", "keras_history": ["batch_normalization_3", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "data_format": "channels_last", "keepdims": false}, "registered_name": null, "name": "global_average_pooling2d", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 6, 6, 256], "dtype": "float32", "keras_history": ["activation_3", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "rate": 0.35, "seed": null, "noise_shape": null}, "registered_name": null, "name": "dropout", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 256], "dtype": "float32", "keras_history": ["global_average_pooling2d", 0, 0]}}], "kwargs": {"training": false}}]}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "quantization_config": null}, "registered_name": null, "build_config": {"input_shape": [null, 256]}, "name": "dense", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 256], "dtype": "float32", "keras_history": ["dropout", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "rate": 0.25, "seed": null, "noise_shape": null}, "registered_name": null, "name": "dropout_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 64], "dtype": "float32", "keras_history": ["dense", 0, 0]}}], "kwargs": {"training": false}}]}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "quantization_config": null}, "registered_name": null, "build_config": {"input_shape": [null, 64]}, "name": "dense_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 64], "dtype": "float32", "keras_history": ["dropout_1", 0, 0]}}], "kwargs": {}}]}], "input_layers": ["input_layer", 0, 0], "output_layers": ["dense_1", 0, 0]}, "registered_name": "Functional", "build_config": {"input_shape": null}, "compile_config": {}}'


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
            / "agua.weights.h5"
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
    print("MODELO: AGUA")
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
        f"{modelo.count_params():,}"
    )

    print()

    modelo.summary()
