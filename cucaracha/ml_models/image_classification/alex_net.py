import keras
from keras import layers

from cucaracha.ml_models.model_architect import ModelArchitect


class AlexNet(ModelArchitect):
    """
    AlexNet is a custom model architecture for image classification tasks,
    inheriting from the ModelArchitect base class. This model is based on the
    original AlexNet architecture, designed to handle large-scale image classification
    tasks with high computational efficiency.

    Reference:
        Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
        ImageNet Classification with Deep Convolutional Neural Networks.
        Advances in Neural Information Processing Systems, 25, 1097-1105.

    Attributes:
        img_shape (tuple): The shape of the input images (height, width).
        num_classes (int): The number of output classes for classification.

    Methods:
        get_model():
            Builds and returns the Keras model based on the AlexNet architecture.
        __str__():
            Returns a string representation of the model, including a summary of the
            model architecture with trainable parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(modality='image_classification', **kwargs)
        self.img_shape = kwargs.get('img_shape')
        self.num_classes = kwargs.get('num_classes')

    def get_model(self):
        input_shape = (self.img_shape[0], self.img_shape[1], 3)
        inputs = keras.Input(shape=input_shape)

        # x = keras.models.Sequential()

        # Entry block
        # Layer 1: Convolutional layer with 64 filters of size 11x11x3
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(
            filters=64,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            activation='relu',
        )(x)

        # Layer 2: Max pooling layer with pool size of 3x3
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
        x = layers.Conv2D(
            filters=192, kernel_size=(5, 5), padding='same', activation='relu'
        )(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = layers.Conv2D(
            filters=384, kernel_size=(3, 3), padding='same', activation='relu'
        )(x)
        x = layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same', activation='relu'
        )(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Layer 6: Fully connected layer with 4096 neurons
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)

        # Layer 7: Fully connected layer with 4096 neurons
        x = layers.Dense(4096, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return keras.Model(inputs, outputs)

    def __str__(self):
        output = super().__str__()
        self.get_model().summary(show_trainable=True)
        return output
