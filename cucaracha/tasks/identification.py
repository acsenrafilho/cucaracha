import numpy as np

from cucaracha.tasks import call_cucacha_image_task


def identify_personal_document(input: np.array, auto_fit: bool = True):
    """
    Identify the personal document type from an image, seting the document 
    type based on the Brazilian personal documents such as CNH, CPF and RG.

    If the document is not identified as CNH, CPF or RG, the method will return
    the string 'others' to exemplify an unrecognized document type.

    Note:
        This method is not intended to be used for document verification, i.e.
        it does not check if the document is valid or not, and also does not
        collect any information from the document. It only identifies the type
        as CNH, CPF or RG.

    Note:
        The method assumed that the input image is taken considering the 
        majority of the image space of being as the document itself. Images 
        with partial document or with a lot of noise may not be correctly 
        identified.

    Info:
        For the `auto_fit` option, If the input image is not consistent to the
        ML model input shape, then the method will fit it before prediction. 
        If the user does not want this behavior, e.g. one may want to already
        provide an input data with the correct shape, then the user should set
        `auto_fit` to `False`. 

    Args:
        input (np.array): An image representing the personal document.
        auto_fit (bool, optional): Fits the input shape to ML model needs. Defaults to True.
    Returns:
        tuple: The predicted document type and extra information.
    """
    return call_cucacha_image_task(input, 'cnh_cpf_rg', auto_fit)


def identify_document_is_signed(input: np.array, auto_fit: bool = True):
    """
    Identify if the document is signed or not from an image.

    Note:
        This method is not intended to be used for document verification, i.e.
        it does not check if the document is valid or not, and also does not
        collect any information from the document. It only verifies whether
        the document presents a signature or not.

    Note:
        The method assumes that the signature is well seen in the image, i.e.
        it should be easily identified by a human eye.

    Info:
        For the `auto_fit` option, If the input image is not consistent to the
        ML model input shape, then the method will fit it before prediction. 
        If the user does not want this behavior, e.g. one may want to already
        provide an input data with the correct shape, then the user should set
        `auto_fit` to `False`. 

    Args:
        input (np.array): An image representing the document with or without a signature.
        auto_fit (bool, optional): Fits the input shape to ML model needs. Defaults to True.

    Returns:
        tuple: The predicted document type and extra information.
    """
    return call_cucacha_image_task(input, 'doc_is_signed', auto_fit)
