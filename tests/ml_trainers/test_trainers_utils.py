import os
import sys

import pytest

import tests.sample_paths as sp
from cucaracha.ml_trainers.utils import (
    _check_dataset_folder,
    _check_dataset_folder_permissions,
    _check_paths,
    load_cucaracha_dataset,
)


def test_load_cucaracha_dataset_success():
    ds_path, annot_json = load_cucaracha_dataset(
        sp.DOC_ML_DATASET_CLASSIFICATION
    )
    assert 'organized_data' in ds_path
    assert all(
        [True for item in ('receipt', 'forms', 'law') if item in ds_path]
    )


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
