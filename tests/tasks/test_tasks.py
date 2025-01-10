import numpy as np
import pytest

from cucaracha.tasks import call_cucacha_image_task


def test_call_cucacha_image_task_with_generic_image():
    # Arrange
    image = np.random.rand(500, 1000, 3)
    expected_output = {'probabilities': 'some-output', 'labels': {}}

    # Act
    result = call_cucacha_image_task(image, doc_preset='doc_is_signed')

    # Assert
    assert isinstance(result[0], str)
    assert isinstance(result[1], dict)
