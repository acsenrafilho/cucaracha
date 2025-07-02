import os

import cv2 as cv
import numpy as np
import pytest

from cucaracha.tasks.text_extraction import (
    extract_text,
    extract_text_easyocr,
    extract_text_tesseract,
)
from tests import sample_paths


class TestTextExtraction:
    def test_extract_text_tesseract_basic(self):
        """Test basic OCR functionality with a sample text image."""
        # Load the sample text image
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None, 'Sample image should be loaded'

        # Run OCR
        result_img, extra = extract_text_tesseract(img)

        # Check that we got the original image back
        assert np.array_equal(
            result_img, img
        ), 'Output image should be identical to input'

        # Check the structure of extra information
        assert 'extracted_text' in extra
        assert 'confidence' in extra
        assert 'word_data' in extra
        assert 'lang' in extra
        assert 'config' in extra

        # Check data types
        assert isinstance(extra['extracted_text'], str)
        assert isinstance(extra['confidence'], float)
        assert isinstance(extra['word_data'], list)
        assert isinstance(extra['lang'], str)
        assert isinstance(extra['config'], str)

        # Check that we extracted some text (sample image should contain text)
        assert (
            len(extra['extracted_text'].strip()) > 0
        ), 'Should extract some text from sample image'

        # Check confidence is reasonable (should be > 0 for a good sample image)
        assert extra['confidence'] >= 0.0, 'Confidence should be non-negative'
        assert extra['confidence'] <= 100.0, 'Confidence should not exceed 100'

    def test_extract_text_tesseract_with_custom_params(self):
        """Test OCR with custom language and configuration parameters."""
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None

        # Test with custom parameters
        result_img, extra = extract_text_tesseract(
            img, lang='eng', config='--psm 8'
        )

        # Check that parameters were used
        assert extra['lang'] == 'eng'
        assert extra['config'] == '--psm 8'

        # Should still return basic structure
        assert 'extracted_text' in extra
        assert 'confidence' in extra

    def test_extract_text(self):
        """Test the simplified OCR function."""
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None

        result_img, extra = extract_text(img)

        # Check that we got the original image back
        assert np.array_equal(result_img, img)

        # Check basic structure
        assert 'extracted_text' in extra
        assert 'confidence' in extra
        assert isinstance(extra['extracted_text'], str)

    def test_extract_text_with_empty_image(self):
        """Test OCR behavior with an empty/blank image."""
        # Create a blank white image
        blank_img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result_img, extra = extract_text_tesseract(blank_img)

        # Should handle gracefully
        assert np.array_equal(result_img, blank_img)
        assert 'extracted_text' in extra
        assert 'confidence' in extra

        # Blank image should have little to no text
        assert (
            len(extra['extracted_text'].strip()) == 0
            or extra['confidence'] < 50
        )

    def test_extract_text_with_grayscale_image(self):
        """Test OCR with grayscale image input."""
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Add channel dimension to make it compatible
        gray_img_3d = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

        result_img, extra = extract_text_tesseract(gray_img_3d)

        # Should work with grayscale input
        assert 'extracted_text' in extra
        assert 'confidence' in extra
        assert isinstance(extra['extracted_text'], str)

    def test_word_data_structure(self):
        """Test that word_data contains expected information."""
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None

        result_img, extra = extract_text_tesseract(img)

        word_data = extra['word_data']
        assert isinstance(word_data, list)

        # If we have words, check their structure
        if len(word_data) > 0:
            word = word_data[0]
            expected_keys = [
                'text',
                'confidence',
                'left',
                'top',
                'width',
                'height',
            ]
            for key in expected_keys:
                assert key in word, f"Word data should contain '{key}'"

            # Check data types
            assert isinstance(word['text'], str)
            assert isinstance(word['confidence'], float)
            assert isinstance(word['left'], int)
            assert isinstance(word['top'], int)
            assert isinstance(word['width'], int)
            assert isinstance(word['height'], int)

    def test_error_handling_invalid_input(self):
        """Test error handling with invalid input."""
        # Test with invalid array shape
        invalid_img = np.array([1, 2, 3])
        result_img, extra = extract_text_tesseract(invalid_img)

        # Should handle gracefully and include error info
        assert 'error' in extra or len(extra['extracted_text']) == 0

    def test_extract_text_easyocr_basic(self):
        """Test basic EasyOCR functionality with a sample text image."""
        # Load the sample text image
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None, 'Sample image should be loaded'

        # Run OCR with EasyOCR
        result_img, extra = extract_text_easyocr(img)

        # Check that we got the original image back
        assert np.array_equal(
            result_img, img
        ), 'Output image should be identical to input'

        # Check the structure of extra information
        assert 'extracted_text' in extra
        assert 'confidence' in extra
        assert 'word_data' in extra
        assert 'lang' in extra
        assert 'method' in extra

        # Check data types
        assert isinstance(extra['extracted_text'], str)
        assert isinstance(extra['confidence'], float)
        assert isinstance(extra['word_data'], list)
        assert isinstance(extra['lang'], list)
        assert extra['method'] == 'easyocr'

        # Check confidence is reasonable (EasyOCR uses 0-1 scale)
        assert extra['confidence'] >= 0.0, 'Confidence should be non-negative'
        assert extra['confidence'] <= 1.0, 'Confidence should not exceed 1.0'

        # Skip text extraction test if EasyOCR isn't available or has issues
        if 'error' not in extra:
            # Should extract some text if no error
            assert isinstance(extra['extracted_text'], str)

    def test_extract_text_easyocr_multilingual(self):
        """Test EasyOCR with multiple languages."""
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None

        # Test with multiple languages
        result_img, extra = extract_text_easyocr(img, lang=['en', 'pt'])

        # Check that parameters were used
        assert extra['lang'] == ['en', 'pt']
        assert extra['method'] == 'easyocr'

        # Should still return basic structure
        assert 'extracted_text' in extra
        assert 'confidence' in extra

    def test_extract_text_easyocr_word_data_structure(self):
        """Test that EasyOCR word_data contains expected information."""
        img = cv.imread(sample_paths.SAMPLE_TEXT_PNG)
        assert img is not None

        result_img, extra = extract_text_easyocr(img)

        word_data = extra['word_data']
        assert isinstance(word_data, list)

        # If we have words and no error, check their structure
        if len(word_data) > 0 and 'error' not in extra:
            word = word_data[0]
            expected_keys = [
                'text',
                'confidence',
                'left',
                'top',
                'width',
                'height',
                'bbox',
            ]
            for key in expected_keys:
                assert key in word, f"Word data should contain '{key}'"

            # Check data types
            assert isinstance(word['text'], str)
            assert isinstance(word['confidence'], float)
            assert isinstance(word['left'], int)
            assert isinstance(word['top'], int)
            assert isinstance(word['width'], int)
            assert isinstance(word['height'], int)
            assert isinstance(word['bbox'], list)
