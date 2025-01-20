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
from cucaracha.ml_models.image_segmentation import UNetXception


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


def test_image_classification_model_resnet_50_str():
    model = ResNet50(img_shape=(128, 128), num_classes=2)
    model_str = str(model)

    assert 'Model Architecture modality: image_classification' in model_str


def test_image_classification_model_densenet_121_str():
    model = DenseNet121(img_shape=(128, 128), num_classes=2)
    model_str = str(model)

    assert 'Model Architecture modality: image_classification' in model_str


def test_image_classification_model_soup_str():
    model = ModelSoup(img_shape=(128, 128), num_classes=2)
    model_str = str(model)

    assert 'Model Architecture modality: image_classification' in model_str


def test_image_classification_model_alex_net_str():
    model = AlexNet(img_shape=(128, 128), num_classes=2)
    model_str = str(model)

    assert 'Model Architecture modality: image_classification' in model_str


def test_image_classification_model_small_xception_str():
    model = SmallXception(img_shape=(128, 128), num_classes=2)
    model_str = str(model)

    assert 'Model Architecture modality: image_classification' in model_str


def test_image_segmentation_model_unet_xception_str():
    model = UNetXception(img_shape=(128, 128), num_classes=2)
    model_str = str(model)

    assert 'Model Architecture modality: image_segmentation' in model_str
