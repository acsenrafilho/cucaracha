import keras
import pytest

from cucaracha.ml_models import VALID_MODALITIES
from cucaracha.ml_models.image_segmentation import UNetXception
from cucaracha.ml_models.model_architect import ModelArchitect
from cucaracha.ml_trainers import ImageSegmentationTrainer
from tests import sample_paths as sp


def test_get_model_returns_Keras_model_success():
    obj = ImageSegmentationTrainer(
        sp.SAMPLE_ML_DATASET_SEGMENTATION, num_classes=2
    )

    assert isinstance(obj.model, keras.Model)


def test_load_dataset_success():
    obj = ImageSegmentationTrainer(
        sp.SAMPLE_ML_DATASET_SEGMENTATION, num_classes=2
    )
    dataset = obj.load_dataset()

    assert dataset['train'] is not None and dataset['val'] is not None


def test_train_model_success():
    obj = ImageSegmentationTrainer(
        sp.SAMPLE_ML_DATASET_SEGMENTATION, num_classes=2
    )
    obj.epochs = 1
    obj.batch_size = 1
    obj.train_model()

    assert obj.model.history.history['acc'][0] > 0.0


def test_image_segmentation_trainer_raises_error_when_architecture_is_not_ModelArchitect_class():
    with pytest.raises(ValueError) as e:
        obj = ImageSegmentationTrainer(
            dataset_path=sp.SAMPLE_ML_DATASET_SEGMENTATION,
            num_classes=2,
            architecture='not_image_classification_architecture',
        )

    assert (
        e.value.args[0]
        == 'The provided architecture is not a valid ModelArchitect instance.'
    )


def test_image_segmentation_trainer_raises_error_when_architecture_is_not_image_classification():
    class AnotherArchitect(ModelArchitect):
        def __init__(self):
            super().__init__(modality='image_keypoint_detection')
            pass

        def get_model(self):
            pass

        def __str__(self):
            return 'ModelArchitect'

    new_arch = AnotherArchitect()

    with pytest.raises(ValueError) as e:
        obj = ImageSegmentationTrainer(
            dataset_path=sp.SAMPLE_ML_DATASET_SEGMENTATION,
            num_classes=2,
            architecture=new_arch,
        )

    assert (
        e.value.args[0]
        == 'The provided modality is not valid for image_segmentation task.'
    )


def test_image_segmentation_trainer_raises_error_modality_out_of_list():
    new_arch = UNetXception(img_shape=(128, 128), num_classes=2)
    new_arch.modality = 'not_valid_architecture'

    with pytest.raises(ValueError) as e:
        obj = ImageSegmentationTrainer(
            dataset_path=sp.SAMPLE_ML_DATASET_SEGMENTATION,
            num_classes=2,
            architecture=new_arch,
        )

    assert (
        e.value.args[0]
        == 'The provided modality is not valid for image_segmentation task.'
    )


def test_model_name_parameter_changed():
    obj = ImageSegmentationTrainer(
        dataset_path=sp.SAMPLE_ML_DATASET_SEGMENTATION,
        num_classes=2,
        model_name='new_model_name',
    )

    assert obj.model_name == 'new_model_name'
