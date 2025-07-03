import os
import shutil
import sys

import cv2 as cv
import numpy as np
import pytesseract

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


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
    # Check if tesseract-ocr is installed on the system
    tesseract_cmd = shutil.which('tesseract')
    if tesseract_cmd is None:
        raise EnvironmentError(
            'Tesseract OCR is not installed or not found in PATH. '
            "Please install it (e.g., 'sudo apt install tesseract-ocr' on Linux, "
            "'brew install tesseract' on macOS) and ensure it's available in your PATH."
        )

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


def extract_text(input: np.ndarray, lang='eng'):
    """Extract text from an image using OCR with optimized default settings.

    This is a simplified text extraction method that uses default OCR settings
    optimized for general document text extraction. This method serves as the
    main text extraction interface that can utilize different OCR backends.

    Args:
        input (np.ndarray): The input image containing text to be extracted.
        lang (str, optional): Language code for OCR. Defaults to 'eng'.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The original input image (unchanged)
            - dict: Dictionary with extracted text information

    """
    return extract_text_tesseract(
        input,
        lang=lang,
        config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
    )


def extract_text_easyocr(input: np.ndarray, lang=['en'], gpu=False):
    """Extract text from an image using EasyOCR.

    This method uses EasyOCR for text extraction, providing an alternative to
    Tesseract. EasyOCR is particularly effective for multilingual text and
    doesn't require system-level installation like Tesseract.

    Note:
        EasyOCR performs well on various text orientations and supports many
        languages out of the box. The first run may take longer as it downloads
        the required models.

    Args:
        input (np.ndarray): The input image containing text to be extracted.
            Can be in color (BGR) or grayscale format.
        lang (list, optional): List of language codes for OCR. Defaults to ['en'].
            Common options: ['en'], ['pt'], ['es'], ['fr'], etc.
            Can also use multiple languages: ['en', 'pt']
        gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
            Requires CUDA-compatible GPU and proper EasyOCR installation.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The original input image (unchanged)
            - dict: Dictionary with extracted information:
                - 'extracted_text': The text content extracted from the image
                - 'confidence': Overall confidence score of the OCR result (0-1)
                - 'word_data': Detailed word-level data with bounding boxes and confidences
                - 'lang': Languages used for OCR
                - 'method': OCR method used ('easyocr')

    Raises:
        ImportError: If EasyOCR is not installed
        Exception: If there's an error during OCR processing
    """
    if not EASYOCR_AVAILABLE:
        extra_info = {
            'extracted_text': '',
            'confidence': 0.0,
            'word_data': [],
            'lang': lang,
            'method': 'easyocr',
            'error': 'EasyOCR is not installed. Please install it with: pip install easyocr',
        }
        return input, extra_info

    try:
        # Initialize EasyOCR reader with timeout protection
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError('EasyOCR initialization timeout')

        # Set a timeout for initialization to avoid hanging in CI environments
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        try:
            reader = easyocr.Reader(lang, gpu=gpu)
            signal.alarm(0)  # Cancel the alarm
        except TimeoutError:
            extra_info = {
                'extracted_text': '',
                'confidence': 0.0,
                'word_data': [],
                'lang': lang,
                'method': 'easyocr',
                'error': 'EasyOCR initialization timeout (possibly downloading models)',
            }
            return input, extra_info

        # Extract text using EasyOCR
        results = reader.readtext(input, detail=1)

        # Process results
        extracted_text = ''
        word_data = []
        confidences = []

        for (bbox, text, confidence) in results:
            extracted_text += text + ' '
            confidences.append(confidence)

            # Convert bbox to left, top, width, height format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            left = int(min(x_coords))
            top = int(min(y_coords))
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))

            word_info = {
                'text': text,
                'confidence': float(confidence),
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'bbox': bbox,  # Original bounding box coordinates
            }
            word_data.append(word_info)

        # Calculate overall confidence
        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        extra_info = {
            'extracted_text': extracted_text.strip(),
            'confidence': overall_confidence,
            'word_data': word_data,
            'lang': lang,
            'method': 'easyocr',
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
            'method': 'easyocr',
            'error': str(e),
        }
        return input, extra_info
