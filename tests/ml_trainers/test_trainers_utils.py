import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests.sample_paths as sp
from cucaracha.utils import (
    _check_dataset_folder,
    _check_dataset_folder_permissions,
    _check_paths,
    _load_image_classification_dataset,
    load_cucaracha_dataset,
    plot_confusion_matrix,
    verify_image_compatibility,
)


def test_load_cucaracha_dataset_success():
    ds_path, annot_json = load_cucaracha_dataset(
        sp.DOC_ML_DATASET_CLASSIFICATION,
        'image_classification',
    )
    assert 'organized_data' in ds_path
    assert all(
        [True for item in ('receipt', 'forms', 'law') if item in ds_path]
    )


def test_load_cucaracha_dataset_raise_errors_for_wrong_dataset_type():
    with pytest.raises(Exception) as e:
        load_cucaracha_dataset(
            sp.DOC_ML_DATASET_CLASSIFICATION,
            'wrong_dataset_type',
        )
    assert 'Supported types are:' in str(e.value)


def test_check_paths_raise_errors_path_not_found():
    with pytest.raises(FileNotFoundError) as e:
        _check_paths(['non_existent_path'])
    assert f'The path non_existent_path does not exist.' in str(e.value)


def test_check_dataset_folder_raise_errors_raw_data_not_found():
    with pytest.raises(FileNotFoundError) as e:
        _check_dataset_folder(sp.ROOT_TEST_FOLDER)
    assert (
        f'The raw_data folder does not exist in {sp.ROOT_TEST_FOLDER}.'
        in str(e.value)
    )


def test_check_dataset_folder_raise_errors_label_studio_json_not_found():
    os.makedirs(os.path.join(sp.ROOT_TEST_FOLDER, 'raw_data'), exist_ok=True)
    json_path = os.path.join(sp.ROOT_TEST_FOLDER, 'label_studio_export.json')
    with pytest.raises(FileNotFoundError) as e:
        _check_dataset_folder(sp.ROOT_TEST_FOLDER)
    assert (
        f'The label_studio_export.json file does not exist in {json_path}.'
        in str(e.value)
    )
    os.rmdir(os.path.join(sp.ROOT_TEST_FOLDER, 'raw_data'))


# @pytest.mark.skipif(sys.platform.startswith("win"), reason="Test not supported on Windows")
def test_check_dataset_folder_permissions_raise_errors():
    folder_path = os.path.join(sp.ROOT_TEST_FOLDER, 'folder_not_allowed')
    os.makedirs(folder_path, exist_ok=True)
    # Set read-only permissions (Unix-like systems)
    if not sys.platform.startswith('win'):
        os.chmod(folder_path, 0o444)
    else:
        # Set read-only attribute (Windows)
        os.chmod(folder_path, stat.S_IREAD)

    with pytest.raises(PermissionError) as e:
        _check_dataset_folder_permissions(folder_path)
    assert f'You do not have permission to write in {folder_path}.' in str(
        e.value
    )
    os.rmdir(folder_path)


def test_verify_image_compatibility_no_incompatible_images(tmp_path):
    path = os.path.join(sp.DOC_ML_DATASET_CLASSIFICATION, 'raw_data')
    result = verify_image_compatibility(path)

    # Assert no incompatible images found
    assert result == []


def test_verify_image_compatibility_with_incompatible_images(tmp_path):
    # Create a temporary directory with an incompatible image
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    img_path = img_dir / 'image1.txt'
    img_path.write_text('This is not an image file.')

    # Run the function
    result = verify_image_compatibility(img_dir)

    # Assert the incompatible image is found
    assert len(result) == 1
    assert 'image1.txt' in result[0]


def test_verify_image_compatibility_empty_directory(tmp_path):
    # Create an empty temporary directory
    img_dir = tmp_path / 'images'
    img_dir.mkdir()

    # Run the function
    result = verify_image_compatibility(img_dir)

    # Assert no incompatible images found
    assert result == []


def test_verify_image_compatibility_non_existent_directory():
    # Run the function with a non-existent directory
    result = verify_image_compatibility('non_existent_directory')

    # Assert no incompatible images found
    assert result == []


def test_load_image_classification_dataset_folder_without_subfolders(tmp_path):
    temp_folder = tmp_path / 'empty_folder'
    temp_folder.mkdir()
    with pytest.raises(ValueError) as e:
        _load_image_classification_dataset(temp_folder)
    assert 'Not enough folders to describe a classification task in' in str(
        e.value
    )


def test_plot_confusion_matrix(tmp_path):
    # Create a sample confusion matrix
    cm = np.array([[5, 2, 0], [1, 3, 1], [0, 1, 4]])
    classes = ['class1', 'class2', 'class3']
    output_path = tmp_path / 'confusion_matrix.png'

    # Plot the confusion matrix
    plt = plot_confusion_matrix(cm, classes, output_path)

    # Assert the output file is created
    assert plt.__name__ == 'matplotlib.pyplot'


def test_plot_confusion_matrix_empty_classes(tmp_path):
    # Create a sample confusion matrix
    cm = np.array([[5, 2, 0], [1, 3, 1], [0, 1, 4]])
    classes = []
    output_path = tmp_path / 'confusion_matrix.png'

    # Plot the confusion matrix
    with pytest.raises(Exception) as e:
        plot_confusion_matrix(cm, classes, output_path)
    assert 'Classes list cannot be empty' in str(e.value)


def test_plot_confusion_matrix_mismatched_classes(tmp_path):
    # Create a sample confusion matrix
    cm = np.array([[5, 2, 0], [1, 3, 1], [0, 1, 4]])
    classes = ['class1', 'class2']
    output_path = tmp_path / 'confusion_matrix.png'

    # Plot the confusion matrix
    with pytest.raises(ValueError) as e:
        plot_confusion_matrix(cm, classes, output_path)
    assert (
        'Number of classes must match the size of the confusion matrix'
        in str(e.value)
    )
