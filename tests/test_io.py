import os

import numpy as np

from cucaracha import io

SAMPLE_TEXT_JPG = (
    '.' + os.sep + 'tests' + os.sep + 'files' + os.sep + 'sample-text-en.jpg'
)


def test_load_document_returns_numpy_array():
    out = io.load_document(doc_path=SAMPLE_TEXT_JPG)
    assert isinstance(out, np.ndarray)
