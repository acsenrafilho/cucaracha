import pytest

from cucaracha.ml_models.model_architect import ModelArchitect
from cucaracha.ml_trainers.ml_pattern import check_architecture_pattern


class DummyModelArchitect(ModelArchitect):
    def __init__(self, modality):
        self.modality = modality

    def get_model(self):
        pass

    def __str__(self):
        return f'Model Architecture modality: {self.modality}'


def test_check_architecture_pattern_valid_architecture():
    kwargs = {'architecture': DummyModelArchitect(modality='classification')}
    try:
        check_architecture_pattern(kwargs, model_type='classification')
    except ValueError:
        pytest.fail('Unexpected ValueError raised')


def test_check_architecture_pattern_invalid_architecture_instance():
    kwargs = {'architecture': 'invalid_instance'}
    with pytest.raises(
        ValueError,
        match='The provided architecture is not a valid ModelArchitect instance.',
    ):
        check_architecture_pattern(kwargs, model_type='classification')


def test_check_architecture_pattern_invalid_modality():
    kwargs = {'architecture': DummyModelArchitect(modality='regression')}
    with pytest.raises(
        ValueError,
        match='The provided modality is not valid for classification task.',
    ):
        check_architecture_pattern(kwargs, model_type='classification')


def test_check_architecture_pattern_invalid_architecture_modality():
    kwargs = {'architecture': DummyModelArchitect(modality='regression')}
    with pytest.raises(
        ValueError,
        match='The provided modality is not valid for classification task.',
    ):
        check_architecture_pattern(kwargs, model_type='classification')
