import pytest

from cucaracha.ml_models import VALID_MODALITIES
from cucaracha.ml_models.model_architect import ModelArchitect


def test_model_architect_raises_error_when_modality_is_not_valid():
    class SomeArchitecture(ModelArchitect):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get_model(self):
            pass

    with pytest.raises(ValueError) as e:
        obj = SomeArchitecture(modality='not_valid_modality')

    assert (
        e.value.args[0]
        == f'Invalid modality. Expected one of {VALID_MODALITIES}, got not_valid_modality'
    )


def test_model_architect_str_method():
    class SomeArchitecture(ModelArchitect):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get_model(self):
            pass

    obj = SomeArchitecture(modality='image_classification')
    assert str(obj) == 'Model Architecture modality: image_classification'
