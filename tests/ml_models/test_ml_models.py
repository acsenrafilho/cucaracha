import os

import pytest

from cucaracha.ml_models import CUCARACHA_MODELS
from cucaracha.ml_models.model_download import (
    download_cucaracha_dataset,
    download_cucaracha_model,
)


def test_download_cucaracha_model_success():
    model_url = CUCARACHA_MODELS['image_classification']['doc_is_signed'][
        'variation'
    ]
    path = download_cucaracha_model(model_url)

    assert path is not None


def test_download_cucaracha_model_invalid_url():
    invalid_url = 'invalid-url.com/model'
    with pytest.raises(ValueError):
        download_cucaracha_model(invalid_url)


def test_download_cucaracha_model_empty_url():
    empty_url = ''
    with pytest.raises(ValueError):
        download_cucaracha_model(empty_url)


def test_download_cucaracha_model_none_url():
    none_url = None
    with pytest.raises(TypeError):
        download_cucaracha_model(none_url)
