import os

from .image_classification_trainer import ImageClassificationTrainer

os.environ['KERAS_BACKEND'] = 'tensorflow'

__all__ = ['ImageClassificationTrainer']
