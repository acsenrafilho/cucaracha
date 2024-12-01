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