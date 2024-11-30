from abc import ABC, abstractmethod

from cucaracha.ml_models import VALID_MODALITIES


class ModelArchitect(ABC):
    def __init__(self, **kwargs):
        self.modality = kwargs.get('modality', None)
        # valid_modalities = ['image_classification', 'image_keypoint_detection', 'image_object_detection']
        if self.modality is None or self.modality not in VALID_MODALITIES:
            raise ValueError(
                f'Invalid modality. Expected one of {VALID_MODALITIES}, got {self.modality}'
            )

    @abstractmethod
    def get_model(self):
        pass

    def __str__(self):
        return f'Model Architecture modality: {self.modality}'
