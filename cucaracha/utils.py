import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cucaracha.ml_models import CUCARACHA_PRESETS


def load_cucaracha_dataset(dataset_path: str, dataset_type: str):
    """
    Load and organize the Cucaracha dataset from the given path. A `cucaracha`
    dataset is generally organized in the following way:
    1. A `raw_data` folder containing all the raw images is no specific order.
    2. A `label_studio_export.json` file containing the dataset annotations. It
    is always named `label_studio_export.json` and it has been exported from
    the Label Studio tool as a JSON file.

    This function performs the following steps:
    1. Loads raw data from the specified dataset path.
    2. Reads the 'label_studio_export.json' file to get dataset annotations.
    3. Copies images using symbolic links to appropriate label folders based on
    annotations.

    Args:
        dataset_path (str): The path to the dataset directory containing 'raw_data' and 'label_studio_export.json'.
    Returns:
        tuple: A tuple containing:
            - train_dataset (str): The path to the organized training dataset.
            - dataset (dict): The full loaded dataset annotations from 'label_studio_export.json'.
    Raises:
        ValueError: If the source path for an image is not found.
    """
    if dataset_type not in CUCARACHA_PRESETS.keys():
        raise ValueError(
            f"Dataset type '{dataset_type}' is not supported. Supported types are: {list(CUCARACHA_PRESETS.keys())}"
        )

    if dataset_type == 'image_classification':
        return _load_image_classification_dataset(dataset_path)
    if dataset_type == 'image_segmentation':
        return _load_image_segmentation_dataset(dataset_path)


def prepare_image_classification_dataset(dataset_path: str, json_data: json):
    # TODO Verify if this function will be public or private... is there an application for this function outside this module?
    label_set = set()
    for item in json_data:
        for annotation in item['annotations'][0]['result']:
            if 'value' in annotation and 'choices' in annotation['value']:
                label_set.update(annotation['value']['choices'])

    for label in label_set:
        label_folder = os.path.join(dataset_path, 'organized_data', label)
        os.makedirs(label_folder, exist_ok=True)

    return label_set


def verify_image_compatibility(dataset_path: str):
    """
    Verify the compatibility of images in a given dataset path with TensorFlow.
    This function traverses through the directory specified by `dataset_path` and checks each image file
    to determine if it is compatible with TensorFlow. If an image is found to be incompatible, its path
    is added to a list of incompatible images, and a message is printed to the console.

    Args:
        dataset_path (str): The path to the dataset directory containing image files.
    Returns:
        List[str]: A list of file paths for images that are incompatible with TensorFlow.
    Example:
        >>> import tests.sample_paths as sp
        >>> import os
        >>> dataset_path = os.path.join(sp.DOC_ML_DATASET_CLASSIFICATION, 'raw_data')
        >>> incompatible_images = verify_image_compatibility(dataset_path)
        >>> len(incompatible_images)
        0
    """

    incompatible_images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not _check_tensorflow_image(file_path):
                incompatible_images.append(file_path)
                print(f'Incompatible image found: {file_path}')
    return incompatible_images


def _check_tensorflow_image(image_path: str):
    """
    Checks if an image can be loaded using TensorFlow.
    This function attempts to read and decode an image from the given file path
    using TensorFlow's I/O and image processing functions. If the image cannot
    be loaded, it raises a ValueError with an appropriate error message.
    Args:
        image_path (str): The file path to the image to be checked.
    Raises:
        ValueError: If the image cannot be loaded by TensorFlow, with details
                    about the encountered error.
    """
    checked = True
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img)
    except Exception as e:
        checked = False
        RuntimeWarning(
            f'The image {image_path} could not be loaded by tensorflow. Error: {e}'
        )
    return checked


def _check_paths(path_list: list):
    for path in path_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f'The path {path} does not exist.')


def _check_dataset_folder(dataset_path: str):
    raw_data_path = os.path.join(dataset_path, 'raw_data')
    json_path = os.path.join(dataset_path, 'label_studio_export.json')

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f'The raw_data folder does not exist in {dataset_path}.'
        )

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f'The label_studio_export.json file does not exist in {json_path}.'
        )


def _check_dataset_folder_permissions(datataset_path: str):
    if not os.access(datataset_path, os.W_OK):
        raise PermissionError(
            f'You do not have permission to write in {datataset_path}.'
        )


def _load_image_classification_dataset(dataset_path: str):
    class_names = []
    train_dataset = dataset_path

    # Assumes there are raw data folder and label studio json file
    raw_data_folder = os.path.join(dataset_path, 'raw_data')
    label_studio_json = os.path.join(dataset_path, 'label_studio_export.json')

    # Check if the dataset is already organized
    if not os.path.exists(raw_data_folder) or not os.path.exists(
        label_studio_json
    ):
        subfolders = sorted(
            [f.path for f in os.scandir(dataset_path) if f.is_dir()]
        )
        if len(subfolders) <= 1:
            raise ValueError(
                f'Not enough folders to describe a classification task in {dataset_path}.'
            )
        class_names = [os.path.basename(folder) for folder in subfolders]

        # Return the dataset path if it is already organized
        return train_dataset, class_names

    # Continue with the organization process
    train_dataset = os.path.join(dataset_path, 'organized_data')

    # Load the cucaracha label_studio_export.json file
    with open(label_studio_json, 'r') as f:
        dataset = json.load(f)

    class_names = prepare_image_classification_dataset(dataset_path, dataset)

    # Copy images to appropriate label folders
    for item in dataset:
        img_filename = item['data']['img'].split(os.sep)[-1]

        src_path = ''
        matching_files = [
            f for f in os.listdir(raw_data_folder) if f in img_filename
        ]
        if matching_files:
            src_path = os.path.join(raw_data_folder, matching_files[0])

        if not src_path or not _check_tensorflow_image(
            src_path
        ):   # pragma: no cover
            RuntimeWarning(
                f'Image path not found or not compatible to tensorflow: {img_filename}. Skipping...'
            )
            continue  # pragma: no cover

        if os.path.exists(src_path):   # pragma: no cover
            try:
                annotation = item['annotations'][0]['result']
                label = annotation[0]['value']['choices'][0]

                # for label in labels:
                dst_path = os.path.join(
                    dataset_path, 'organized_data', label, matching_files[0]
                )
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
            except IndexError as e:   # pragma: no cover
                Warning(
                    f'Annotation does not found to file {src_path} Warning: {e}'
                )
                continue

    return train_dataset, class_names


def _load_image_segmentation_dataset(dataset_path: str):
    # Load images
    img_folder = os.path.join(dataset_path, 'images')

    # Load annotations
    ann_folder = os.path.join(dataset_path, 'annotations')

    # Merge the list of images with corresponding annotation file
    img_files = [
        f
        for f in os.listdir(img_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    ann_files = [
        f
        for f in os.listdir(ann_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Ensure that each annotation file has a corresponding image file
    matched_img_files = []
    matched_ann_files = []

    for ann_file in ann_files:
        img_file = next(
            (
                img
                for img in img_files
                if os.path.splitext(img)[0] == os.path.splitext(ann_file)[0]
            ),
            None,
        )
        if img_file:
            matched_img_files.append(img_file)
            matched_ann_files.append(ann_file)

    img_files = matched_img_files
    ann_files = matched_ann_files

    if len(img_files) != len(ann_files):
        raise ValueError('The number of images and annotations do not match.')

    dataset = []
    for img_file, ann_file in zip(img_files, ann_files):
        img_path = os.path.join(img_folder, img_file)
        ann_path = os.path.join(ann_folder, ann_file)

        if not _check_tensorflow_image(img_path):
            RuntimeWarning(
                f'Incompatible image found: {img_path}. Skipping...'
            )
            continue

        dataset.append((img_path, ann_path))

    return dataset


def plot_confusion_matrix(
    cm, target_names, title='Confusion matrix', cmap=None, normalize=True
):
    """
    Generates a plot for a given confusion matrix.

    This function takes a confusion matrix from sklearn and generates a visual
    representation using matplotlib. It can display either raw numbers or
    normalized proportions.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix from sklearn.metrics.confusion_matrix.

    target_names : list of str
        List of class names corresponding to the labels in the confusion matrix.
        For example: ['high', 'medium', 'low'] or [0, 1, 2]

    title : str, optional, default='Confusion matrix'
        The text to display at the top of the matrix as a title for the plot.

    cmap : matplotlib colormap, optional, default=None
        Colormap to be used for the plot. If None, defaults to plt.cm.Blues.

    normalize : bool, optional, default=True
        If True, the confusion matrix will be normalized to show proportions.
        If False, the raw numbers will be displayed.

    Returns
    -------
    plt : matplotlib.pyplot
        The plot object for the confusion matrix.

    plot_confusion_matrix(cm=cm,                  # confusion matrix created by
                          normalize=True,         # show proportions
                          target_names=y_labels_vals,  # list of names of the classes
                          title=best_estimator_name)   # title of graph

    References
    ----------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if len(target_names) == 0:
        raise ValueError('Classes list cannot be empty')

    if len(target_names) != cm.shape[0]:
        raise ValueError(
            'Number of classes must match the size of the confusion matrix'
        )

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                '{:0.4f}'.format(cm[i, j]),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black',
            )
        else:
            plt.text(
                j,
                i,
                '{:,}'.format(cm[i, j]),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black',
            )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(
        'Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass
        )
    )

    return plt
