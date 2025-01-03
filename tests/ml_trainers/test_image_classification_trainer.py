import keras
import pytest

from cucaracha.ml_models import VALID_MODALITIES
from cucaracha.ml_models.image_classification.small_xception import (
    SmallXception,
)
from cucaracha.ml_models.model_architect import ModelArchitect
from cucaracha.ml_trainers import ImageClassificationTrainer
from tests import sample_paths as sp


def test_load_dataset_success():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    dataset = obj.load_dataset()

    assert dataset['train'] is not None and dataset['val'] is not None


def test_get_model_returns_Keras_model_success():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 4)

    assert isinstance(obj.model, keras.Model)


def test_train_model_success():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    obj.epochs = 1
    obj.batch_size = 1
    obj.train_model()

    assert obj.model.history.history['acc'][0] > 0.0


def test_image_classification_trainer_raises_error_when_architecture_is_not_ModelArchitect_class():
    with pytest.raises(ValueError) as e:
        obj = ImageClassificationTrainer(
            dataset_path=sp.DOC_ML_DATASET_CLASSIFICATION,
            num_classes=3,
            architecture='not_image_classification_architecture',
        )

    assert (
        e.value.args[0]
        == 'The provided architecture is not a valid ModelArchitect instance.'
    )


def test_image_classification_trainer_raises_error_when_architecture_is_not_image_classification():
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
        obj = ImageClassificationTrainer(
            dataset_path=sp.DOC_ML_DATASET_CLASSIFICATION,
            num_classes=3,
            architecture=new_arch,
        )

    assert (
        e.value.args[0]
        == 'The provided modality is not valid for image_classification task.'
    )


def test_image_classification_trainer_raises_error_modality_out_of_list():
    with pytest.raises(ValueError) as e:
        obj = ImageClassificationTrainer(
            dataset_path=sp.DOC_ML_DATASET_CLASSIFICATION,
            num_classes=3,
            modality='not_valid_architecture',
        )

    assert (
        e.value.args[0]
        == f'The provided modality is not valid for image_classification task.'
    )


def test_image_classification_trainer_raises_error_modality_out_of_list():
    new_arch = SmallXception(img_shape=(128, 128), num_classes=3)
    new_arch.modality = 'not_valid_architecture'

    with pytest.raises(ValueError) as e:
        obj = ImageClassificationTrainer(
            dataset_path=sp.DOC_ML_DATASET_CLASSIFICATION,
            num_classes=3,
            architecture=new_arch,
        )

    assert (
        e.value.args[0]
        == 'The provided modality is not valid for image_classification task.'
    )


def test_model_name_parameter_changed():
    obj = ImageClassificationTrainer(
        dataset_path=sp.DOC_ML_DATASET_CLASSIFICATION,
        num_classes=3,
        model_name='new_model_name',
    )

    assert obj.model_name == 'new_model_name'
