import numpy as np

from cucaracha.tasks import call_cucacha_image_task


def identify_personal_document(input: np.array, auto_fit: bool = True):
    return call_cucacha_image_task(input, 'cnh_cpf_rg', auto_fit)


def identify_document_is_signed(input: np.array, auto_fit: bool = True):
    return call_cucacha_image_task(input, 'doc_is_signed', auto_fit)
