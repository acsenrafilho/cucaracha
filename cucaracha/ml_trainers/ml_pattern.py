import os
from abc import ABC, abstractmethod

from cucaracha.ml_models.model_architect import ModelArchitect
from cucaracha.ml_trainers.utils import (
    _check_dataset_folder,
    _check_dataset_folder_permissions,
    _check_paths,
)


class MLPattern(ABC):
    def __init__(self, dataset_path: str):  # pragma: no cover
        # _check_paths([dataset_path])
        self.dataset_path = os.path.abspath(dataset_path)
        self.batch_size = 64
        self.epochs = 500

    @abstractmethod
    def load_dataset(self):   # pragma: no cover
        # _check_dataset_folder(self.dataset_path)
        _check_dataset_folder_permissions(self.dataset_path)
        pass

    @abstractmethod
    def train_model(self):   # pragma: no cover
        pass


def check_architecture_pattern(kwargs: dict, model_type=str):
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

    if (
        kwargs.get('architecture')
        and kwargs['architecture'].modality != model_type
    ):
        raise ValueError(
            f'The provided architecture is not an {model_type} Architect instance.'
        )
