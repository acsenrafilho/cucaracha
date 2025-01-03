import keras
from keras import layers

from cucaracha.ml_models.model_architect import ModelArchitect


class UNetXception(ModelArchitect):
    """
    UNetXception is a deep learning model for image segmentation that combines the
    architecture of U-Net with the Xception model's depthwise separable convolutions.

    Reference:
        François Chollet. "Xception: Deep Learning with Depthwise Separable Convolutions."
        arXiv preprint arXiv:1610.02357 (2017).

    **kwargs: Arbitrary keyword arguments. Expected keys are:

        - img_shape (tuple): Shape of the input images (height, width).

        - num_classes (int): Number of classes for the segmentation task.

    Methods:
        get_model():
            Builds and returns the UNetXception model.

        __str__():
            Returns a string representation of the model, including a summary of the model architecture.
    """

    def __init__(self, **kwargs):
        super().__init__(modality='image_segmentation', **kwargs)
        self.img_shape = kwargs.get('img_shape')
        self.num_classes = kwargs.get('num_classes')

    def get_model(self):
        inputs = keras.Input(shape=self.img_shape + (3,))

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding='same')(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation('relu')(x)
            x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation('relu')(x)
            x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding='same')(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(
            self.num_classes, 3, activation='softmax', padding='same'
        )(x)

        # Define the model
        return keras.Model(inputs, outputs)

    def __str__(self):
        super().__str__()
        self.get_model().summary(show_trainable=True)
