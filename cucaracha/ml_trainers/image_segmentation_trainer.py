import datetime
import os

import keras
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

from cucaracha.ml_models.image_segmentation import UNetXception
from cucaracha.ml_trainers.ml_pattern import (
    MLPattern,
    check_architecture_pattern,
)
from cucaracha.utils import load_cucaracha_dataset


class ImageSegmentationTrainer(MLPattern):
    def __init__(self, dataset_path: str, **kwargs):
        """
        This is the main constructor for a general Image Segmentation ML method.

        Note:
            The `dataset_path` should follow the `cucaracha` dataset folder
            organization. More details about how to organize the dataset can be
        """
        super().__init__(dataset_path)
        check_architecture_pattern(kwargs, 'image_segmentation')

        self.img_shape = kwargs.get('img_shape', (160, 160))
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 500)
        self.num_classes = kwargs.get('num_classes', 2)
        self.model = None

        self.architecture = kwargs.get('architecture', None)
        # If no architecture is provided, use the default one
        if kwargs.get('architecture') is None:
            default = UNetXception(
                img_shape=self.img_shape, num_classes=self.num_classes
            )
            self.architecture = default
            self.model = default.get_model()
        else:
            self.architecture = kwargs['architecture']
            self.model = self.architecture.get_model()

        self.dataset = self.load_dataset()

        self.loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.metrics = [keras.metrics.SparseCategoricalAccuracy(name='acc')]
        self.optmizer = keras.optimizers.Adam(1e-4)

        # Define the default model name to save
        time = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')
        ds_name = os.path.basename(os.path.normpath(self.dataset_path))
        modality = self.architecture.modality
        self.model_name = (
            f'mod-{modality}-dataset-{ds_name}-timestamp-{time}.keras'
        )
        if 'model_name' in kwargs:
            self.model_name = kwargs['model_name']

    def load_dataset(self):
        super().load_dataset()

        # Prepare all the dataset environment
        # Create subfolders for each label
        dataset_path = load_cucaracha_dataset(
            self.dataset_path, 'image_segmentation'
        )

        def load_img_masks(
            input_img_path, target_img_path
        ):  # pragma: no cover
            input_img = tf_io.read_file(input_img_path)
            input_img = tf_io.decode_png(input_img, channels=3)
            input_img = tf_image.resize(input_img, self.img_shape)
            input_img = tf_image.convert_image_dtype(input_img, 'float32')

            target_img = tf_io.read_file(target_img_path)
            target_img = tf_io.decode_png(target_img, channels=1)
            target_img = tf_image.resize(
                target_img, self.img_shape, method='nearest'
            )
            target_img = tf_image.convert_image_dtype(target_img, 'uint8')

            # # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # target_img -= 1
            return input_img, target_img

        # For faster debugging, limit the size of data
        # if max_dataset_len:
        #     input_img_paths = input_img_paths[:max_dataset_len]
        #     target_img_paths = target_img_paths[:max_dataset_len]

        split_size = int(0.8 * len(dataset_path))
        input_img_paths = [path[0] for path in dataset_path[:split_size]]
        target_img_paths = [path[1] for path in dataset_path[:split_size]]

        train_dataset = tf_data.Dataset.from_tensor_slices(
            (input_img_paths, target_img_paths)
        )
        train_dataset = train_dataset.map(
            load_img_masks, num_parallel_calls=tf_data.AUTOTUNE
        )

        input_img_paths = [path[0] for path in dataset_path[split_size:]]
        target_img_paths = [path[1] for path in dataset_path[split_size:]]

        valid_dataset = tf_data.Dataset.from_tensor_slices(
            (input_img_paths, target_img_paths)
        )
        valid_dataset = valid_dataset.map(
            load_img_masks, num_parallel_calls=tf_data.AUTOTUNE
        )

        return {
            'train': train_dataset.batch(self.batch_size),
            'val': valid_dataset.batch(self.batch_size),
        }

    def train_model(self, callbacks: list = None):
        if not callbacks:
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    os.path.join(self.dataset_path, self.model_name),
                    save_best_only=True,
                )
            ]

        self.model.compile(
            optimizer=self.optmizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        self.model.fit(
            self.dataset['train'],
            epochs=self.epochs,
            callbacks=callbacks,
            batch_size=self.batch_size,
            validation_data=self.dataset['val'],
        )
