# Text Extraction

This module provides OCR (Optical Character Recognition) functionality for extracting text from document images. It supports multiple OCR backends including Tesseract and EasyOCR, allowing users to choose the best method for their specific use case.

All methods follow the standard pattern of being image filters, taking an image as input and returning both the original image and extracted text information.

## Available Methods

### `extract_text(input, lang='eng')`
The main text extraction interface with optimized default settings. Currently uses Tesseract OCR backend with character whitelisting for better accuracy on document text.

**Example Usage:**
```python
import cv2 as cv
from cucaracha import extract_text

# Load an image containing text
img = cv.imread('document.png')

# Extract text with default settings
result_img, extra = extract_text(img)

# Access the extracted text
text = extra['extracted_text']
confidence = extra['confidence']

print(f"Extracted text: {text}")
print(f"Confidence: {confidence:.2f}%")
```

**Expected Output Format:**
```python
{
    'extracted_text': 'This is the text found in the image',
    'confidence': 85.5,  # 0-100% confidence score
    'word_data': [
        {
            'text': 'This',
            'confidence': 89.2,
            'left': 45, 'top': 12, 'width': 35, 'height': 18
        },
        # ... more words
    ],
    'lang': 'eng',
    'config': '--psm 6 -c tessedit_char_whitelist=...'
}
```

### `extract_text_tesseract(input, lang='eng', config='--psm 6')`
Full-featured Tesseract OCR with customizable parameters for advanced users who need specific OCR configurations.

**Example Usage:**
```python
from cucaracha import extract_text_tesseract

# OCR with custom configuration for single words
result_img, extra = extract_text_tesseract(
    img, 
    lang='eng', 
    config='--psm 8'  # Single word mode
)

# OCR with Portuguese language support
result_img, extra = extract_text_tesseract(
    img, 
    lang='por'
)
```

### `extract_text_easyocr(input, lang=['en'], gpu=False)`
Alternative OCR method using EasyOCR, which is particularly effective for multilingual text and doesn't require system-level Tesseract installation.

**Example Usage:**
```python
from cucaracha import extract_text_easyocr

# Basic EasyOCR text extraction
result_img, extra = extract_text_easyocr(img)

# Multilingual text extraction
result_img, extra = extract_text_easyocr(
    img, 
    lang=['en', 'pt', 'es']  # English, Portuguese, Spanish
)

# With GPU acceleration (requires CUDA)
result_img, extra = extract_text_easyocr(img, gpu=True)
```

**Expected Output Format:**
```python
{
    'extracted_text': 'This is the text found in the image',
    'confidence': 0.85,  # 0-1.0 confidence score
    'word_data': [
        {
            'text': 'This',
            'confidence': 0.89,
            'left': 45, 'top': 12, 'width': 35, 'height': 18,
            'bbox': [[45, 12], [80, 12], [80, 30], [45, 30]]
        },
        # ... more words
    ],
    'lang': ['en'],
    'method': 'easyocr'
}
```

## Integration with Document Processing

The OCR methods work seamlessly with the library's Document class and preprocessing methods:

```python
from cucaracha import Document, extract_text, otsu

# Load a PDF document
doc = Document('multi_page_document.pdf')

# Process first page
page = doc.get_page(0)

# Apply preprocessing for better OCR results
otsu_img, _ = otsu(page)  # Apply Otsu thresholding

# Extract text from preprocessed image
result_img, extra = extract_text(otsu_img)

print(f"Extracted text: {extra['extracted_text']}")
print(f"Number of words found: {len(extra['word_data'])}")
```

## Tips for Better OCR Results

1. **Preprocessing**: Apply image preprocessing methods like Otsu thresholding or noise removal before OCR for better accuracy.

2. **Method Selection**: 
   - Use `extract_text()` for general document text extraction
   - Use `extract_text_tesseract()` when you need specific Tesseract configurations
   - Use `extract_text_easyocr()` for multilingual documents or when Tesseract installation is problematic

3. **Language Support**: Always specify the correct language for better accuracy:
   - English: `'eng'` (Tesseract) or `['en']` (EasyOCR)
   - Portuguese: `'por'` (Tesseract) or `['pt']` (EasyOCR)
   - Spanish: `'spa'` (Tesseract) or `['es']` (EasyOCR)

4. **Confidence Filtering**: Use confidence scores to filter out low-quality text extractions:
   ```python
   result_img, extra = extract_text(img)
   if extra['confidence'] > 70:  # Only accept high-confidence results
       text = extra['extracted_text']
   ```

::: tasks.text_extraction