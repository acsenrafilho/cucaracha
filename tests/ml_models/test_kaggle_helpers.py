import os

import pytest

from cucaracha.ml_models import CUCARACHA_PRESETS
from cucaracha.ml_models.kaggle_helpers import (
    collect_cucaracha_model,
    download_cucaracha_dataset,
    download_cucaracha_model,
)


def test_download_cucaracha_model_success():
    model_url = CUCARACHA_PRESETS['image_classification']['doc_is_signed'][
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


def test_download_cucaracha_dataset_success():
    dataset_url = CUCARACHA_PRESETS['image_classification']['doc_is_signed']
    path = download_cucaracha_dataset(dataset_url['dataset'])

    assert path is not None


def test_download_cucaracha_dataset_raise_error_wrong_url():
    wrong_url = 'wrong-url.com/dataset'
    with pytest.raises(ValueError):
        download_cucaracha_dataset(wrong_url)


def test_collect_cucaracha_model_bad_preset():
    bad_preset = 'non_existent_preset'
    with pytest.raises(Exception):
        collect_cucaracha_model(bad_preset)


def test_collect_cucaracha_model_with_preset_from_image_classification():
    preset = 'doc_is_signed'
    model = collect_cucaracha_model(preset)

    assert model is not None


def test_collect_cucaracha_model_output_structure():
    preset = 'doc_is_signed'
    output = collect_cucaracha_model(preset)

    assert isinstance(output, dict)
    assert 'model_path' in output
    assert 'labels' in output
    assert output['model_path'] is not None
    assert isinstance(output['labels'], dict)
    assert 'signed' in list(output['labels'].values())
