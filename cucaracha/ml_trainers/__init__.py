import os

from .image_classification_trainer import ImageClassificationTrainer
from .image_segmentation_trainer import ImageSegmentationTrainer

os.environ['KERAS_BACKEND'] = 'tensorflow'

__all__ = ['ImageClassificationTrainer', 'ImageSegmentationTrainer']
