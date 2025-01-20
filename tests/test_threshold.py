import warnings

import numpy as np
import pytest

from cucaracha import Document
from cucaracha.tasks.threshold import binary_threshold, otsu
from tests import sample_paths


@pytest.mark.parametrize(
    'in_path',
    [
        (sample_paths.SAMPLE_TEXT_JPG),
        (sample_paths.SAMPLE_TEXT_PNG),
        (sample_paths.SAMPLE_TEXT_TIF),
        (sample_paths.SAMPLE_TEXT_PDF),
    ],
)
def test_otsu_binarization_with_default_images(in_path):
    obj = Document(in_path)
    out, _ = otsu(obj.get_page(0))

    assert np.min(out) == 0
    assert np.max(out) == 255


@pytest.mark.parametrize(
    'in_path',
    [
        (sample_paths.SAMPLE_TEXT_JPG),
        (sample_paths.SAMPLE_TEXT_PNG),
        (sample_paths.SAMPLE_TEXT_TIF),
        (sample_paths.SAMPLE_TEXT_PDF),
    ],
)
def test_otsu_show_warning_with_non_gray_scale_image(in_path):
    obj = Document(in_path)
    _ = otsu(obj.get_page(0))
    warnings.warn(
        RuntimeWarning(
            'Input image is not gray-scaled pixel. The method converted automatically.'
        )
    )


@pytest.mark.parametrize(
    'in_path',
    [
        (sample_paths.SAMPLE_TEXT_JPG),
        (sample_paths.SAMPLE_TEXT_PNG),
        (sample_paths.SAMPLE_TEXT_TIF),
        (sample_paths.SAMPLE_TEXT_PDF),
    ],
)
def test_binary_binarization_with_default_images(in_path):
    obj = Document(in_path)
    out, _ = binary_threshold(obj.get_page(0), 160)

    assert np.min(out) == 0
    assert np.max(out) == 255


@pytest.mark.parametrize(
    'in_path',
    [
        (sample_paths.SAMPLE_TEXT_JPG),
        (sample_paths.SAMPLE_TEXT_PNG),
        (sample_paths.SAMPLE_TEXT_TIF),
        (sample_paths.SAMPLE_TEXT_PDF),
    ],
)
def test_binary_show_warning_with_non_gray_scale_image(in_path):
    obj = Document(in_path)
    _ = binary_threshold(obj.get_page(0), 160)
    warnings.warn(
        RuntimeWarning(
            'Input image is not gray-scaled pixel. The method converted automatically.'
        )
    )
