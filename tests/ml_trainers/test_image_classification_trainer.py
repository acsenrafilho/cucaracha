import keras
import numpy as np
import pytest
import tensorflow as tf

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


def test_load_dataset_success_with_organized_folder():
    obj = ImageClassificationTrainer(
        sp.DOC_ML_DATASET_CLASSIFICATION_ORGANIZED, 3
    )
    dataset = obj.load_dataset()

    assert dataset['train'] is not None and dataset['val'] is not None


def test_load_dataset_execute_without_data_augmention():
    obj = ImageClassificationTrainer(
        sp.DOC_ML_DATASET_CLASSIFICATION_ORGANIZED,
        3,
        use_data_augmentation=False,
    )

    assert obj.dataset is not None


def test_get_model_returns_Keras_model_success():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 4)

    assert isinstance(obj.model, keras.Model)


def test_train_model_success():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    obj.epochs = 1
    obj.batch_size = 1
    obj.train_model()

    assert obj.model.history.history['acc'][0] > 0.0


def test_batch_size_parameter_changed():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    assert obj.batch_size == 64
    obj.batch_size = 16

    assert obj.batch_size == 16


def test_batch_size_parameter_changed_from_constructor():
    obj = ImageClassificationTrainer(
        sp.DOC_ML_DATASET_CLASSIFICATION, 3, batch_size=16
    )
    assert obj.batch_size == 16


def test_epochs_parameter_changed():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    assert obj.epochs == 500
    obj.epochs = 10

    assert obj.epochs == 10


def test_epochs_parameter_changed_from_constructor():
    obj = ImageClassificationTrainer(
        sp.DOC_ML_DATASET_CLASSIFICATION, 3, epochs=10
    )
    assert obj.epochs == 10


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


def test_class_names_collected_successfully():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    obj.load_dataset()

    assert obj.class_names is not None
    assert len(obj.class_names) == 3
    assert 'law' in list(obj.class_names.values())
    assert 'receipt' in list(obj.class_names.values())
    assert 'form' in list(obj.class_names.values())


def test_data_augmentation_default_applied():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    img = np.random.rand(128, 128, 3)
    output = obj.data_generator(img)

    assert output.shape == (128, 128, 3)
    assert np.abs(np.subtract(output, img)).all() > 0


def test_data_augmentation_custom_applied():
    custom_layers = [
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(0.2),
    ]
    obj = ImageClassificationTrainer(
        sp.DOC_ML_DATASET_CLASSIFICATION, 3, data_generator=custom_layers
    )
    img = np.random.rand(1, 128, 128, 3)
    output = obj.data_generator(img)

    assert output.shape == (1, 128, 128, 3)
    assert np.abs(np.subtract(output, img)).all() > 0


def test_data_augmentation_invalid_input():
    invalid_data_generator = 'invalid_data_generator'
    with pytest.raises(Exception) as e:
        obj = ImageClassificationTrainer(
            sp.DOC_ML_DATASET_CLASSIFICATION,
            3,
            data_generator=invalid_data_generator,
        )

    assert str(e.value) == 'Data generator must be a list of Keras layers.'


def test_model_compilation_using_user_options():
    obj = ImageClassificationTrainer(
        sp.DOC_ML_DATASET_CLASSIFICATION,
        3,
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['rmsprop'],
    )

    assert isinstance(obj.optimizer, str) and obj.optimizer == 'sgd'
    assert isinstance(obj.loss, str) and obj.loss == 'binary_crossentropy'
    assert isinstance(obj.metrics, list) and obj.metrics[0] == 'rmsprop'


def test_model_training_history():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    obj.epochs = 1
    obj.batch_size = 1
    obj.train_model()

    assert obj.history is not None
    assert 'acc' in obj.history.history
    assert 'val_acc' in obj.history.history


def test_model_training_history_with_custom_callbacks():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)
    obj.epochs = 1
    obj.batch_size = 1
    obj.train_model(callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

    assert obj.history is not None
    assert 'acc' in obj.history.history
    assert 'val_acc' in obj.history.history


def test_class_weights_calculated():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 3)

    assert obj.class_weights is not None
    assert len(obj.class_weights.values()) == 3
    assert all(weight > 0 for weight in obj.class_weights.values())
