import os
import random

import keras
import tensorflow as tf

from cucaracha.ml_models.image_classification import SmallXception
from cucaracha.ml_trainer.ml_pattern import MLPattern
from cucaracha.ml_trainer.utils import (
    load_cucaracha_dataset,
    prepare_image_classification_dataset,
)


class ImageClassificationTrainer(MLPattern):
    def __init__(self, dataset_path: str, num_classes: int, **kwargs):
        super().__init__(dataset_path)
        self.num_classes = num_classes
        self.img_shape = kwargs.get('img_shape', (128, 128))
        self.model = kwargs.get('architecture')

        self.loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.optmizer = keras.optimizers.Adam(1e-4)
        self.metrics = [keras.metrics.CategoricalAccuracy(name='acc')]

        self.dataset = self.load_dataset()

        # If no architecture is provided, use the default one
        if self.model is None:
            default = SmallXception(
                img_shape=self.img_shape, num_classes=self.num_classes
            )
            self.model = default.get_model()

        if (
            kwargs.get('architecture')
            and kwargs['architecture'].get('modality')
            != 'image_classification'
        ):
            raise ValueError(
                'The provided architecture is not for image classification tasks.'
            )

    def load_dataset(self):
        super().load_dataset()

        train_dataset, dataset = load_cucaracha_dataset(self.dataset_path)

        # Prepare all the dataset environment
        # Create subfolders for each label
        class_names = prepare_image_classification_dataset(
            self.dataset_path, dataset
        )

        # Load the organized data using keras.utils.image_dataset_from_directory
        train_ds, val_ds = keras.utils.image_dataset_from_directory(
            train_dataset,
            class_names=class_names,
            image_size=self.img_shape,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='both',
            seed=random.randint(0, 10000),
        )

        num_classes = len(class_names)
        train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
        val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

        return {'train': train_ds, 'val': val_ds}

    def train_model(self, callbacks: list = None):
        if not callbacks:
            callbacks = [
                keras.callbacks.ModelCheckpoint(f'save_at_{self.epochs}.keras') # TODO Fix this path using dataset_path
            ]

        # model = self.get_model(self.num_classes)

        self.model.compile(
            optimizer=self.optmizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        # keras.utils.image_dataset_from_directory()
        # TODO Verify to usage of data_augmentation directly in fit method (see: https://keras.io/examples/vision/image_classification_from_scratch/)
        self.model.fit(
            self.dataset['train'],
            epochs=self.epochs,
            callbacks=callbacks,
            batch_size=self.batch_size,
            validation_data=self.dataset['val'],
        )

    def __str__(self):
        return print(self.model)
