import os

import cv2 as cv
import numpy as np


def otsu(input: np.ndarray):
    """Otsu binarization algoritm

    This method is able to split the image color into a binary representation
    ,i.e. zero and 255 (`int`).

    This is particularly helpful to increase text constrast and help to assist
    other algorithms, such as OCR and others

    Note:
        This method performas an automatic binarization. However, it is
        important to use on image that has a reasonable text vs background
        contrast. Otherwise, the results can overlay some chunks of the image
        and poorly remove part of the text.


    Examples:
        >>> input_img = cv.imread('.'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.png')
        >>> output_img, extra = otsu(input_img)
        >>> np.min(output_img)
        0
        >>> np.max(output_img)
        255
        >>> output_img.shape
        (320, 320)
        >>> extra
        {'thr_value': 160.0}

    Warning:
        The Otsu method is based on gray-scaled images. Then, if the input
        data is not at gray-scale format, the method makes an automatic
        convertion using OpenCV (`cvtColor()`) method. It is assumed a
        `RBG2GRAY_SCALE` flag to do it. If you want to have more control to the
        data convertion to gray-scale, than make it before using this method

    Args:
        input (np.ndarray): The input image that will be binarized using Otsu
        method. It must be a gray-scale data. If not provided, then it is done
        an automatic conversion.

    Reference:
        Nobuyuki Otsu, A Threshold Selection Method from Gray-Level Histograms. IEEE
        Transactions on Systems, Man, and Cybernetics ( Volume: 9, Issue: 1,
        January 1979)

    Returns:
        (np.array, dict): The output image in binary format. It is also give the threshold value found by the Otsu method (key: `thr_value`)
    """
    in_process = input
    if input.shape[2] > 1:
        in_process = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        RuntimeWarning(
            'Input image is not gray-scaled pixel. The method converted automatically.'
        )

    thr, output = cv.threshold(
        in_process, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    extra_info = {'thr_value': thr}
    return output, extra_info


def binary_threshold(input: np.ndarray, thr: int):
    """A binary threshold method setting by manual adjustment.

    Here, the user that select the value that must be used to make the binary
    threshold.

    The method applied here is the following:

    - Values above the threshold value will be placed to `maximum` (255)

    - Values bellow the threshold value will be placed to `minimum` (0)

    Note:
        Any value in allowed. However, the value must be in the image data
        range.

    Args:
        input (np.ndarray): The input image that will be binarized
        thr (int): The threshold value

    Returns:
        (np.ndarray): The output image after the binarization
    """
    in_process = input
    if input.shape[2] > 1:
        in_process = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        RuntimeWarning(
            'Input image is not gray-scaled pixel. The method converted automatically.'
        )

    thr, output = cv.threshold(in_process, thr, 255, cv.THRESH_BINARY)

    extra_info = {'thr_value': thr}
    return output, extra_info
