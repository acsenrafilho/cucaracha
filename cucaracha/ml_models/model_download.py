import kagglehub

from cucaracha.ml_models import CUCARACHA_MODELS


def download_cucaracha_model(model_url: str):
    """
    Downloads a Cucaracha model from the given URL.
    This function checks if the provided model URL is present in the CUCARACHA_MODELS dictionary.
    If the URL is valid, it attempts to download the model using the kagglehub library.

    The downloaded files are located in the home/.cache folder.

    Note:
        We used the kagglehub library to make all the operations here. The
        CUCARACHA_MODELS dictionary is expected to have a nested structure where
        the model variations are stored under a 'variation' key. If the URL is
        valid, it attempts to download the model using the kagglehub library.

    Args:
        model_url (str): The URL of the model to be downloaded. Must be a valid kagglehub input.
    Returns:
        str: The path where the model is downloaded.
    Raises:
        ValueError: If the model URL is not present in CUCARACHA_MODELS or if there is an error during download.
    """

    for url in CUCARACHA_MODELS.values():
        for item in url.values():
            if not model_url in item['variation']:
                raise ValueError(
                    f'Model URL {model_url} is not present in CUCARACHA_MODELS'
                )

    try:
        path = kagglehub.model_download(model_url)
    except Exception as e:
        raise ValueError(f'Error downloading the model: {e}')

    return path


def download_cucaracha_dataset(dataset_url: str):
    """
    Downloads the Cucaracha dataset from the given URL using kagglehub.

    The downloaded files are located in the home/.cache folder.

    Note:
        We used the kagglehub library to make all the operations here. The
        CUCARACHA_MODELS dictionary is expected to have a nested structure where
        the model variations are stored under a 'variation' key. If the URL is
        valid, it attempts to download the model using the kagglehub library.


    Args:
        dataset_url (str): The URL of the dataset to be downloaded.
    Returns:
        str: The path where the dataset is downloaded.
    Raises:
        ValueError: If there is an error during the download process.
    """

    try:
        path = kagglehub.dataset_download(dataset_url)
    except Exception as e:
        raise ValueError(f'Error downloading the dataset: {e}')

    return path
