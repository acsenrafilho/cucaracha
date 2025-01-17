from abc import ABC, abstractmethod

from cucaracha.ml_models import VALID_MODALITIES


class ModelArchitect(ABC):
    """
    Abstract base class for defining model architectures.
    Attributes:
        modality (str): The modality of the model architecture. Expected values are defined in VALID_MODALITIES.
    Methods:
        get_model():
            Abstract method to be implemented by subclasses to return the model architecture.
        __str__():
            Returns a string representation of the model architecture, including its modality.
    """

    def __init__(self, **kwargs):
        self.modality = kwargs.get('modality', None)
        if self.modality is None or self.modality not in VALID_MODALITIES:
            raise ValueError(
                f'Invalid modality. Expected one of {VALID_MODALITIES}, got {self.modality}'
            )

    @abstractmethod
    def get_model(self):   # pragma: no cover
        pass

    def __str__(self):   # pragma: no cover
        return f'Model Architecture modality: {self.modality}'
