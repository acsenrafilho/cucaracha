import cv2 as cv
import numpy as np
import pytest

from cucaracha.tasks.identification import (
    identify_document_is_signed,
    identify_personal_document,
)
from tests import sample_paths as sp


def test_identify_document_is_signed_with_valid_preset():
    input_data = np.random.rand(100, 100, 3)
    result = identify_document_is_signed(input_data)
    assert result is not None


def test_identify_document_is_signed_with_empty_input():
    input_data = np.array([])
    with pytest.raises(Exception) as e:
        identify_document_is_signed(input_data)

    assert 'Input data cannot be empty.' in e.value.args[0]


def test_identify_document_is_signed_with_none_input():
    input_data = None
    with pytest.raises(Exception) as e:
        identify_document_is_signed(input_data)

    assert 'Input data cannot be None.' in e.value.args[0]


def test_identify_document_is_signed_success():
    input_data = np.random.rand(100, 100, 3)
    result, _ = identify_document_is_signed(input_data)

    assert result in ['signed', 'not_signed']


def test_identify_personal_document_with_valid_preset():
    input_data = cv.imread(sp.SAMPLE_CNH_JPG)
    doc_preset = 'cnh_cpf_rg'
    result = identify_personal_document(input_data, doc_preset)
    assert result is not None


# def test_identify_personal_document_success_using_cnh():
#     input_data = cv.imread(sp.SAMPLE_CNH_JPG)
#     result = identify_personal_document(input_data)
#     assert result[0] == 'cnh'


# def test_identify_personal_document_success_using_rg():
#     input_data = cv.imread(sp.SAMPLE_RG_JPG)
#     result = identify_personal_document(input_data)
#     assert result[0] == 'rg'


# def test_identify_personal_document_success_using_cpf():
#     input_data = cv.imread(sp.SAMPLE_CPF_JPG)
#     result = identify_personal_document(input_data)
#     assert result[0] == 'cpf'
