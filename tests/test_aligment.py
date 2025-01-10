import cv2 as cv
import numpy as np
import pytest

from cucaracha import Document
from cucaracha.tasks.aligment import inplane_deskew
from tests import sample_paths


@pytest.mark.parametrize(
    'input_rot,input_orig',
    [
        (
            sample_paths.SAMPLE_CUCARACHA_RGB_ROT_MINUS10_PNG,
            sample_paths.SAMPLE_CUCARACHA_RGB_PNG,
        ),
        (
            sample_paths.SAMPLE_CUCARACHA_GRAY_ROT_MINUS10_TIF,
            sample_paths.SAMPLE_CUCARACHA_GRAY_TIF,
        ),
    ],
)
def test_inplane_deskew_correct_images_with_negative_angle(
    input_rot, input_orig
):
    doc = Document(input_rot)
    output, extra = inplane_deskew(doc.get_page(0))
    orig_img = cv.imread(input_orig)

    assert np.abs(np.mean(orig_img) - np.mean(output)) < 50
    assert extra.get('angle') != 0


@pytest.mark.parametrize(
    'input_rot,input_orig',
    [
        (
            sample_paths.SAMPLE_CUCARACHA_RGB_ROT_PLUS10_PNG,
            sample_paths.SAMPLE_CUCARACHA_RGB_PNG,
        ),
        (
            sample_paths.SAMPLE_CUCARACHA_RGB_ROT_PLUS90_PNG,
            sample_paths.SAMPLE_CUCARACHA_RGB_PNG,
        ),
        (
            sample_paths.SAMPLE_CUCARACHA_GRAY_ROT_PLUS10_TIF,
            sample_paths.SAMPLE_CUCARACHA_GRAY_TIF,
        ),
        (
            sample_paths.SAMPLE_CUCARACHA_GRAY_ROT_PLUS90_TIF,
            sample_paths.SAMPLE_CUCARACHA_GRAY_TIF,
        ),
    ],
)
def test_inplane_deskew_correct_images_with_positive_angle(
    input_rot, input_orig
):
    doc = Document(input_rot)
    output, extra = inplane_deskew(doc.get_page(0))
    orig_img = cv.imread(input_orig)

    assert np.abs(np.mean(orig_img) - np.mean(output)) < 50
    assert extra.get('angle') != 0
