import numpy as np
import pytest

from cucaracha import Document
from cucaracha.noise_removal import sparse_dots
from tests import sample_paths


@pytest.mark.parametrize(
    'input,kernel',
    [
        (sample_paths.SAMPLE_TEXT_JPG, 2),
        (sample_paths.SAMPLE_TEXT_PNG, 4),
        (sample_paths.SAMPLE_TEXT_PDF, 8),
        (sample_paths.SAMPLE_TEXT_TIF, 14),
    ],
)
def test_sparse_dots_raise_error_when_kernel_size_is_even(input, kernel):
    doc = Document(input)
    with pytest.raises(Exception) as e:
        sparse_dots(input, kernel)


def test_sparse_dots_remove_major_salt_paper_noise_correctly():
    input = sample_paths.SAMPLE_CUCARACHA_GRAY_SALT_PEPPER_PNG
    doc = Document(input)
    out = sparse_dots(doc.get_page(0))

    assert np.abs(np.mean(out) - np.mean(doc.get_page(0))) < 10
