import os
import warnings

import cv2 as cv
import numpy as np
from tensorflow import keras

from cucaracha.ml_models import CUCARACHA_PRESETS
from cucaracha.ml_models.kaggle_helpers import collect_cucaracha_model

CLASSIFICATION_PRESETS = list(CUCARACHA_PRESETS['image_classification'].keys())


def call_cucacha_image_task(
    input: np.ndarray, doc_preset: str = 'cnh_cpf_rg', auto_fit: bool = True
):
    _check_input(input)
    _check_doc_preset(doc_preset)

    model_info = collect_cucaracha_model(doc_preset)
    model_files = [
        f for f in os.listdir(model_info['model_path']) if f.endswith('.keras')
    ]
    if not model_files:
        raise FileNotFoundError(
            f"No .keras file found in {model_info['model_path']}"
        )
    model_path = os.path.join(model_info['model_path'], model_files[0])

    # Load the model and labels
    model = keras.models.load_model(model_path)
    labels = model_info['labels']

    # Prepare the input image to the model input layer
    in_model_shape = model.input_shape[1:]  # Exclude the batch size
    if input.shape != in_model_shape:
        warnings.warn(
            f'Warning: Input shape {input.shape} is different from the model input shape {in_model_shape}.'
        )
        if auto_fit:
            warnings.warn(
                f'Warning: Auto-fitting the input image to the model input shape {in_model_shape}.'
            )
            input_image = cv.resize(
                input, (in_model_shape[1], in_model_shape[0])
            )
            input_image = np.expand_dims(input_image, axis=0)
        else:
            raise ValueError(
                f'Input shape {input.shape} does not match the model input shape {in_model_shape}.'
            )

    prediction = model.predict(input_image)
    prediction_label = labels[np.argmax(prediction)]

    extra = {'probabilities': prediction, 'labels': labels}
    return prediction_label, extra


def _check_doc_preset(doc_preset: str):
    if doc_preset not in CLASSIFICATION_PRESETS:
        raise ValueError(
            f'Invalid document preset {doc_preset}. Supported presets are: {CLASSIFICATION_PRESETS}'
        )


def _check_input(input: np.ndarray):
    if input is None:
        raise TypeError('Input data cannot be None.')
    if input.size == 0:
        raise ValueError('Input data cannot be empty.')
