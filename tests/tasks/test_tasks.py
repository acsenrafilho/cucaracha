import numpy as np
import pytest

from cucaracha.tasks import _check_doc_preset, call_cucacha_image_task


def test_call_cucacha_image_task_with_generic_image():
    # Arrange
    image = np.random.rand(500, 1000, 3)
    expected_output = {'probabilities': 'some-output', 'labels': {}}

    # Act
    result = call_cucacha_image_task(image, doc_preset='doc_is_signed')

    # Assert
    assert isinstance(result[0], str)
    assert isinstance(result[1], dict)


def test_call_cucacha_image_task_with_empty_image():
    # Arrange
    image = np.array([])
    # Act
    with pytest.raises(Exception) as e:
        call_cucacha_image_task(image, doc_preset='doc_is_signed')

    assert 'Input data cannot be empty.' in e.value.args[0]


def test_call_cucacha_image_task_with_invalid_image():
    # Arrange
    image = None

    # Act
    with pytest.raises(Exception) as e:
        call_cucacha_image_task(image, doc_preset='doc_is_signed')

    assert 'Input data cannot be None.' in e.value.args[0]


def test_call_cucacha_image_task_with_different_preset():
    # Arrange
    image = np.random.rand(500, 1000, 3)

    # Act
    result = call_cucacha_image_task(image, doc_preset='cnh_cpf_rg')

    # Assert
    assert isinstance(result[0], str)
    assert isinstance(result[1], dict)


def test_call_cucacha_image_task_raise_error_with_input_shape_not_matching_model_input_shape():
    # Arrange
    image = np.random.rand(500, 1000, 3)

    # Act
    with pytest.raises(Exception) as e:
        call_cucacha_image_task(
            image, doc_preset='doc_is_signed', auto_fit=False
        )

    assert 'does not match the model input shape' in e.value.args[0]


def test_check_doc_preset_with_valid_preset():
    # Arrange
    preset = 'doc_is_signed'

    # Act
    result = _check_doc_preset(preset)

    # Assert
    assert result is None


def test_check_doc_preset_with_invalid_preset():
    # Arrange
    preset = 'invalid_preset'

    # Act
    with pytest.raises(ValueError) as e:
        _check_doc_preset(preset)

    # Assert
    assert 'Invalid document preset' in e.value.args[0]


def test_check_doc_preset_with_empty_preset():
    # Arrange
    preset = ''

    # Act
    with pytest.raises(ValueError) as e:
        _check_doc_preset(preset)

    # Assert
    assert 'Invalid document preset' in e.value.args[0]


def test_check_doc_preset_with_none_preset():
    # Arrange
    preset = None

    # Act
    with pytest.raises(ValueError) as e:
        _check_doc_preset(preset)

    # Assert
    assert 'Invalid document preset' in e.value.args[0]
