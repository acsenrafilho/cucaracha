import json
import os


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

        if not src_path:
            raise ValueError(
                f'Source path not found for image: {img_filename}'
            )

        if os.path.exists(src_path):
            annotation = item['annotations'][0]['result']
            label = annotation[0]['value']['choices'][0]

            # for label in labels:
            dst_path = os.path.join(
                dataset_path, 'organized_data', label, matching_files[0]
            )
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

    return train_dataset, class_names


def prepare_image_classification_dataset(dataset_path: str, json_data: json):
    class_names = []
    label_set = set()
    for item in json_data:
        for annotation in item['annotations'][0]['result']:
            if 'value' in annotation and 'choices' in annotation['value']:
                label_set.update(annotation['value']['choices'])

    for label in label_set:
        label_folder = os.path.join(dataset_path, 'organized_data', label)
        class_names.append(label)
        os.makedirs(label_folder, exist_ok=True)

    return class_names


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
