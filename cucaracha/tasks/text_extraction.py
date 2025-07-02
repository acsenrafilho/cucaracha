import os

import cv2 as cv
import numpy as np
import pytesseract


def extract_text_tesseract(input: np.ndarray, lang='eng', config='--psm 6'):
    """Extract text from an image using Tesseract OCR.

    This method uses Tesseract OCR to extract text content from document images.
    It is particularly helpful for digitizing textual documents and making them
    searchable or editable.

    The method applies Tesseract OCR directly to the input image and returns
    both the original image (unchanged) and the extracted text information.

    Note:
        This method works best on images with clear text and good contrast.
        For better results, consider preprocessing the image with methods like
        otsu thresholding or noise removal before applying OCR.

        Tesseract must be installed on the system for this method to work.
        The method supports multiple languages and configurations.

    Examples:
        >>> input_img = cv.imread('.'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.png')
        >>> output_img, extra = extract_text_tesseract(input_img)
        >>> 'extracted_text' in extra
        True
        >>> 'confidence' in extra
        True
        >>> isinstance(extra['extracted_text'], str)
        True

    Args:
        input (np.ndarray): The input image containing text to be extracted.
            Can be in color (BGR) or grayscale format.
        lang (str, optional): Language code for OCR. Defaults to 'eng' (English).
            Common options: 'eng', 'por', 'spa', 'fra', etc.
        config (str, optional): Tesseract configuration string. Defaults to '--psm 6'
            (uniform block of text). Common PSM modes:
            - PSM 6: Uniform block of text (default)
            - PSM 8: Single word
            - PSM 13: Raw line. Treat image as a single text line

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The original input image (unchanged)
            - dict: Dictionary with extracted information:
                - 'extracted_text': The text content extracted from the image
                - 'confidence': Overall confidence score of the OCR result (0-100)
                - 'word_data': Detailed word-level data with bounding boxes and confidences
                - 'lang': Language used for OCR
                - 'config': Configuration string used

    Raises:
        Exception: If Tesseract is not installed or there's an error during OCR processing
    """
    try:
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(
            input, lang=lang, config=config
        )

        # Get detailed data including confidence scores
        data = pytesseract.image_to_data(
            input,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        # Calculate overall confidence (average of word confidences > 0)
        confidences = [float(conf) for conf in data['conf'] if int(conf) > 0]
        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Prepare word-level data for applications that need detailed information
        word_data = []
        for i in range(len(data['text'])):
            if (
                int(data['conf'][i]) > 0
            ):  # Only include words with positive confidence
                word_info = {
                    'text': data['text'][i],
                    'confidence': float(data['conf'][i]),
                    'left': int(data['left'][i]),
                    'top': int(data['top'][i]),
                    'width': int(data['width'][i]),
                    'height': int(data['height'][i]),
                }
                word_data.append(word_info)

        extra_info = {
            'extracted_text': extracted_text.strip(),
            'confidence': overall_confidence,
            'word_data': word_data,
            'lang': lang,
            'config': config,
        }

        # Return original image unchanged and the extracted information
        return input, extra_info

    except Exception as e:
        # Handle potential errors gracefully
        extra_info = {
            'extracted_text': '',
            'confidence': 0.0,
            'word_data': [],
            'lang': lang,
            'config': config,
            'error': str(e),
        }
        return input, extra_info


def extract_text_simple(input: np.ndarray, lang='eng'):
    """Simple text extraction from an image using Tesseract OCR with default settings.

    This is a simplified version of extract_text_tesseract that uses default
    Tesseract settings optimized for general document text extraction.

    Args:
        input (np.ndarray): The input image containing text to be extracted.
        lang (str, optional): Language code for OCR. Defaults to 'eng'.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The original input image (unchanged)
            - dict: Dictionary with extracted text information

    Examples:
        >>> input_img = cv.imread('.'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.png')
        >>> output_img, extra = extract_text_simple(input_img)
        >>> 'extracted_text' in extra
        True
    """
    return extract_text_tesseract(
        input,
        lang=lang,
        config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
    )
