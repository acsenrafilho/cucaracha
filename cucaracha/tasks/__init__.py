import os

import cv2 as cv
import numpy as np
from tensorflow import keras

from cucaracha.ml_models import CUCARACHA_PRESETS
from cucaracha.ml_models.kaggle_helpers import collect_cucaracha_model
from cucaracha.utils import image_auto_fit

CLASSIFICATION_PRESETS = list(CUCARACHA_PRESETS['image_classification'].keys())


def call_cucacha_image_task(
    input: np.ndarray, doc_preset: str = 'cnh_cpf_rg', auto_fit: bool = True
):
    """
    Processes an input image using a pre-trained model specified by the document preset.

    Note:
        This method is directly oriented to image classification tasks. To see
        what the presets availble to be used in this method, check the
        `cucaracha.ml_models.CUCARACHA_PRESETS` variable and the list of
        `image_classification` keys.

    Info:
        For the `auto_fit` option, If the input image is not consistent to the
        ML model input shape, then the method will fit it before prediction.
        If the user does not want this behavior, e.g. one may want to already
        provide an input data with the correct shape, then the user should set
        `auto_fit` to `False`.

    Args:
        input (np.ndarray): The image to be used in the ML model.
        doc_preset (str, optional): Cucaracha preset to be used. Defaults to 'cnh_cpf_rg'.
        auto_fit (bool, optional): Fits the input shape to ML model needs. Defaults to True.

    Raises:
        FileNotFoundError: If the preset is not located in the cucaracha models
        ValueError: Input shape does not match the model input shape. Only raised when `auto_fit` is False.

    Returns:
        tuple: The predicted label and extra information.
    """
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
    in_model_shape = model.input_shape[1:]
    labels = model_info['labels']

    # Prepare the input image to the model input layer
    if input.shape != in_model_shape:
        if not auto_fit:
            raise ValueError(
                f'Input shape {input.shape} does not match the model input shape {in_model_shape}.'
            )
    input_image = image_auto_fit(input, in_model_shape)

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
