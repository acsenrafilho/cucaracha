import os
from abc import ABC, abstractmethod

from cucaracha.ml_trainers.utils import (
    _check_dataset_folder,
    _check_dataset_folder_permissions,
    _check_paths,
)


class MLPattern(ABC):
    def __init__(self, dataset_path: str):  # pragma: no cover
        _check_paths([dataset_path])
        self.dataset_path = os.path.abspath(dataset_path)
        self.batch_size = 64
        self.epochs = 500

    @abstractmethod
    def load_dataset(self):   # pragma: no cover
        _check_dataset_folder(self.dataset_path)
        _check_dataset_folder_permissions(self.dataset_path)
        pass

    @abstractmethod
    def train_model(self):   # pragma: no cover
        pass
