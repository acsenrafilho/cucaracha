import keras

from cucaracha.ml_trainer import ImageClassificationTrainer
from tests import sample_paths as sp


def test_load_dataset_success():
    obj = ImageClassificationTrainer(sp.DOC_ML_DATASET_CLASSIFICATION, 4)
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

    assert obj.model.history.history['acc'][0] > 0
