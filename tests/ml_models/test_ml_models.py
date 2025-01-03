import os

import pytest

from cucaracha.ml_models import CUCARACHA_PRESETS
from cucaracha.ml_models.image_classification import (
    AlexNet,
    DenseNet121,
    GoogleLeNet,
    ModelSoup,
    ResNet50,
    SmallXception,
)
from cucaracha.ml_models.model_download import (
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


def test_image_classification_model_small_xception_build_success():
    model = SmallXception(img_shape=(128, 128), num_classes=2)
    model.get_model()

    assert model is not None


def test_image_classification_model_alex_net_build_success():
    model = AlexNet(img_shape=(128, 128), num_classes=2)
    model.get_model()

    assert model is not None


def test_image_classification_model_google_le_net_build_success():
    model = GoogleLeNet(img_shape=(128, 128), num_classes=2)
    model.get_model()

    assert model is not None


def test_image_classification_model_resnet_50_build_success():
    model = ResNet50(img_shape=(128, 128), num_classes=2)
    model.get_model()

    assert model is not None


def test_image_classification_model_densenet_121_build_success():
    model = DenseNet121(img_shape=(128, 128), num_classes=2)
    model.get_model()

    assert model is not None


def test_image_classification_model_soup_build_success():
    model = ModelSoup(img_shape=(128, 128), num_classes=2)
    model.get_model()

    assert model is not None
