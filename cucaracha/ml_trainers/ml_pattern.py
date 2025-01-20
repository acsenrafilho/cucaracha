import os
from abc import ABC, abstractmethod

from cucaracha.ml_models.model_architect import ModelArchitect
from cucaracha.utils import _check_dataset_folder_permissions, _check_paths


class MLPattern(ABC):
    """
    Abstract class to define the pattern for the ML training process.

    Args:
        ABC (_type_): Abstract class to define the pattern for the ML training process.
    """

    def __init__(self, dataset_path: str):  # pragma: no cover
        _check_paths([dataset_path])
        self.dataset_path = os.path.abspath(dataset_path)
        self.batch_size = 64
        self.epochs = 500

    @abstractmethod
    def load_dataset(self):   # pragma: no cover
        _check_dataset_folder_permissions(self.dataset_path)
        pass

    @abstractmethod
    def train_model(self):   # pragma: no cover
        pass


def check_architecture_pattern(kwargs: dict, model_type=str):
    """
    Check if the provided architecture is a valid ModelArchitect instance.

    This method only evaluates whether the provided architecture is a valid
    ModelArchitect instance, which means that it has the required attributes
    and methods to be used in the training process.

    Args:
        kwargs (dict): The keys that are configured in the ML architecture.
        model_type (_type_, optional): One of the valid ML model type. See the VALID_MODALITIES preset at the cucaracha.ml_models init file.

    Raises:
        ValueError: If the 'architecture' key is provided and it is not a valid ModelArchitect instance.
        ValueError: If the 'architecture' key is provided and the modality is not valid for the model_type task.
        ValueError: If the 'architecture' key is provided and it is not an {model_type} Architect instance.
    """
    if kwargs.get('architecture') and not isinstance(
        kwargs.get('architecture'), ModelArchitect
    ):
        raise ValueError(
            'The provided architecture is not a valid ModelArchitect instance.'
        )
    if (
        kwargs.get('architecture')
        and kwargs.get('architecture').modality != model_type
    ):
        raise ValueError(
            f'The provided modality is not valid for {model_type} task.'
        )
