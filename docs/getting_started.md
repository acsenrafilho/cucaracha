# Getting Started

Welcome to the Cucaracha library! This guide will help you get started with using the library to process documents, including creating a `Document` object, selecting pages, and converting PDFs or images to numpy arrays.

## Introduction

The Cucaracha library is designed to simplify the process of working with documents, whether they are PDFs or images. It provides a straightforward way to load, manipulate, and analyze document pages.

## Creating a Document Object

To begin using the Cucaracha library, you first need to create a `Document` object. This object serves as the main interface for interacting with your documents.

```python
from cucaracha import Document

# Create a Document object from a PDF file
doc = Document('path/to/your/document.pdf')

# Alternatively, create a Document object from an image file
doc = Document('path/to/your/image.jpg')
```

In this example, replace `'path/to/your/document.pdf'` and `'path/to/your/image.jpg'` with the actual paths to your PDF or image files.

!!! tip "Automatic Conversion from PDF/Images to Numpy Array"
    The Cucaracha library can automatically convert pages from PDFs or images into numpy arrays. This is particularly useful for image processing and analysis tasks.

## Basic Functionalities

### Selecting a Page

Once you have created a Document object, you can select a specific page to work with. This is useful when dealing with multi-page PDFs.

```python
# Select the first page of the document
page = doc.select_page(0)
```

In this example, `0` refers to the first page of the document. You can change the index to select different pages.

### General Usage

Here is a complete example that demonstrates how to create a Document object, select a page, and convert it to a numpy array:

```python
from cucaracha import Document
from cucaracha.threshold import otsu
from tests import sample_paths

# Here we can use a sample PDF document located at sample_paths
obj = Document(sample_paths.SAMPLE_TEXT_PDF)

# After the document is loaded, it can be exported the numpy array using using
# the get_page() method
page = obj.get_page(0)

# A simple application can be using otsu algorithm
out, extra = otsu(page)

# The resulting image (out) is a binary unsigned 8-bits image
np.min(out) == 0
np.max(out) == 255
```

This example provides a general idea of how to use the Cucaracha library to work with documents. You can extend this basic functionality to suit your specific needs, such as processing multiple pages or performing image analysis.

### DPI Estimation

The Cucaracha library includes automatic DPI estimation when DPI information is not available in document headers. This is particularly useful for scanned documents or images without embedded resolution metadata.

#### Basic Usage

```python
from cucaracha import Document

# Basic DPI estimation
doc = Document('./sample-document.pdf')
estimated_dpi = doc.estimate_dpi()
print(f"Estimated DPI: {estimated_dpi}")

# Check current DPI and document dimensions
current_dpi = doc.get_metadata('resolution')['resolution']
page_shape = doc.get_page(0).shape
print(f"Current DPI: {current_dpi}")
print(f"Document shape: {page_shape}")
```

#### Estimation Methods

The DPI estimation supports multiple methods, each suited for different scenarios:

```python
# Auto method - intelligently selects best approach based on file type
auto_dpi = doc.estimate_dpi(method='auto')

# PDF dimensions method - uses internal PDF page dimensions (PDF files only)
pdf_dpi = doc.estimate_dpi(method='pdf_dimensions')

# Page size method - assumes A4 page size (210 × 297 mm)
a4_dpi = doc.estimate_dpi(method='page_size')

print(f"Auto method: {auto_dpi}")
print(f"PDF dimensions method: {pdf_dpi}")
print(f"Page size method: {a4_dpi}")
```

#### Automatic Estimation During Loading

For convenience, you can enable automatic DPI estimation when creating a Document object:

```python
# Automatically estimate and apply DPI during document loading
doc_with_estimation = Document('./sample-document.pdf', estimate_dpi=True)
estimated_resolution = doc_with_estimation.get_metadata('resolution')['resolution']
print(f"DPI with estimation enabled: {estimated_resolution}")
print(f"Document shape: {doc_with_estimation.get_page(0).shape}")
```

#### Working with Different File Types

The estimation approach varies depending on the file type:

```python
# PDF files - uses internal document dimensions for accuracy
pdf_doc = Document('./document.pdf')
pdf_estimated_dpi = pdf_doc.estimate_dpi()
print(f"PDF DPI estimation: {pdf_estimated_dpi}")

# Image files - uses A4 size assumptions
img_doc = Document('./scanned-page.jpg')
img_estimated_dpi = img_doc.estimate_dpi()
print(f"Image DPI estimation: {img_estimated_dpi}")
print(f"Image shape: {img_doc.get_page(0).shape}")
```

#### Resolution Impact Comparison

Different DPI settings significantly affect the output resolution. Here's how various DPI values impact document rendering:

```python
# Compare different resolution settings
doc_96 = Document('./sample-document.pdf', resolution=96)
doc_150 = Document('./sample-document.pdf', resolution=150)  
doc_300 = Document('./sample-document.pdf', resolution=300)

print(f"96 DPI shape:  {doc_96.get_page(0).shape}")
print(f"150 DPI shape: {doc_150.get_page(0).shape}")
print(f"300 DPI shape: {doc_300.get_page(0).shape}")
```

Higher DPI values result in larger image dimensions, providing more detail but requiring more memory and processing time.

#### Estimation Methods Overview

- **Auto method**: Automatically chooses the best approach based on file type (PDF dimensions for PDFs, page size for images)
- **PDF dimensions**: Uses internal PDF page dimensions for accurate DPI calculation (PDF files only)
- **Page size**: Estimates based on A4 page size assumptions (210 × 297 mm)

!!! note "DPI Estimation Accuracy"
    DPI estimation works best with full-page documents. For document excerpts or non-standard page sizes, the library provides reasonable defaults and helpful warnings when assumptions may not apply. The estimation includes sanity checking that validates estimates and provides warnings for unrealistic values (outside 50-600 DPI range).

!!! note "Many extension possibilities"
    There are many other applications and algorithms that can be used with the numpy array exposed image (from the obj.get_page() method). Examples can be found in libraries such as OpenCV, SimpleITK, Scikit-Image, Seaborn, Matplotlib, and many others. These libraries offer a wide range of tools for image processing, analysis, and visualization, allowing you to extend the capabilities of the Cucaracha library to meet your specific needs.

!!! info "`cucaracha` often has an `extra` help"
    The "extra" output in the Cucaracha image processing methods provides additional information about the processing results. This can include metadata, processing parameters, or intermediate results that can be useful for further analysis or debugging. For example, when using the otsu method, the "extra" output might contain the threshold value used for binarization. This additional information can help you understand the processing steps and even apply it to other reasoning

The Cucaracha library is a powerful tool for working with documents, offering easy-to-use functionalities for loading, selecting, and converting document pages. Whether you are dealing with PDFs or images, the library provides a seamless way to handle your documents and prepare them for further analysis.

We hope this guide helps you get started with the Cucaracha library. Happy document processing! 


## Using Deep Learning for your needs

To use the `ml_trainers` and `ml_models` modules in the Cucaracha library for machine learning adjustments tailored to your specific applications, follow the steps below:

### Using `ml_trainers` and `ml_models` for ML Adjustments

Here they are some steps to create or adjust and `cucaracha` ML model:

1. Importing the Necessary Modules
First, import the necessary modules from the Cucaracha library:

```python
from cucaracha.ml_models.image_classification.small_xception import SmallXception
from cucaracha.ml_trainers.image_classification_trainer import ImageClassificationTrainer
```

2. Setting Up the Dataset
Ensure your dataset is organized according to the Cucaracha dataset folder structure. You can find more details about organizing your dataset in the [documentation](contribute.md).

3. Initializing the Model
Create an instance of the SmallXception model architecture:

```python
model_architecture = SmallXception(img_shape=(128, 128), num_classes=3)
```

4. Initializing the Trainer
Create an instance of the ImageClassificationTrainer with the dataset path and the number of classes:

```python
trainer = ImageClassificationTrainer(
    dataset_path='path/to/your/dataset',
    num_classes=3,
    architecture=model_architecture
)
```

5. Loading the Dataset
Load the dataset using the load_dataset method:

```python
dataset = trainer.load_dataset()
```

!!! note
    The call of `load_dataset()` is automatically made when an `ml_trainers` class is instanciated. However, if you want to evaluate the training dataset directly, you can used the direct call as given at the example to obtain the training data at your hands.

6. Training the Model
Train the model using the train_model method. You can also provide custom callbacks if needed:

```python
trainer.epochs = 10
trainer.batch_size = 32
trainer.train_model()
```

!!! note
    Here also the `epochs` and `batch_size` are defined automatically by the `ml_trainers` class in use. This example simple shows that you can change it before the `train_model()` execution. 

!!! tip
    Others Keras implementations can be used here, for example Keras Callbacks.

7. Saving the Model
After training, save the model:

```python
trainer.model.save('path/to/save/your_model.keras')
```

Here is a complete example:

```python
from cucaracha.ml_models.image_classification.small_xception import SmallXception
from cucaracha.ml_trainers.image_classification_trainer import ImageClassificationTrainer

# Initialize the model architecture
model_architecture = SmallXception(img_shape=(128, 128), num_classes=3)

# Initialize the trainer
trainer = ImageClassificationTrainer(
    dataset_path='path/to/your/dataset',
    num_classes=3,
    architecture=model_architecture
)

# Load the dataset
dataset = trainer.load_dataset()

# Train the model
trainer.epochs = 10
trainer.batch_size = 32
trainer.train_model()

# Save the model
trainer.model.save('path/to/save/your_model.keras')
```

Additional Information:

- The `ml_trainers` module provides essential methodologies for training machine learning models tailored to specific modalities.
- The `ml_models` module includes various model architectures that can be used for different machine learning tasks.