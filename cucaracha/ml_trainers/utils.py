import json
import os

import tensorflow as tf


def load_cucaracha_dataset(dataset_path: str):
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

    # Load raw data, json file and create organized data
    raw_data_folder = os.path.join(dataset_path, 'raw_data')
    label_studio_json = os.path.join(dataset_path, 'label_studio_export.json')
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

        if not src_path or not _check_tensorflow_image(src_path):
            RuntimeWarning(
                f'Image path not found or not compatible to tensorflow: {img_filename}. Skipping...'
            )
            continue

        if os.path.exists(src_path):
            try:
                annotation = item['annotations'][0]['result']
                label = annotation[0]['value']['choices'][0]

                # for label in labels:
                dst_path = os.path.join(
                    dataset_path, 'organized_data', label, matching_files[0]
                )
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
            except IndexError as e:
                Warning(
                    f'Annotation does not found to file {src_path} Warning: {e}'
                )
                continue

    return train_dataset, class_names


def prepare_image_classification_dataset(dataset_path: str, json_data: json):
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
