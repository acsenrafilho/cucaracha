import keras
from keras import layers
from keras.layers import concatenate

from cucaracha.ml_models.model_architect import ModelArchitect


class GoogleLeNet(ModelArchitect):
    """
    GoogleLeNet is a custom model architecture for image classification tasks,
    inheriting from the ModelArchitect base class. This model is based on the
    GoogleLeNet (Inception v1) architecture, designed to handle large-scale image
    classification tasks with high computational efficiency.

    Reference:
        Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015).
        Going deeper with convolutions.
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1-9.

    Attributes:
        img_shape (tuple): The shape of the input images (height, width).
        num_classes (int): The number of output classes for classification.

    Methods:
        get_model():
            Builds and returns the Keras model based on the GoogleLeNet architecture.
        __str__():
            Returns a string representation of the model, including a summary of the
            model architecture with trainable parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(modality='image_classification', **kwargs)
        self.img_shape = kwargs.get('img_shape')
        self.num_classes = kwargs.get('num_classes')

    def get_model(self):
        # input layer
        input_shape = (self.img_shape[0], self.img_shape[1], 3)
        input_layer = keras.Input(shape=input_shape)

        # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
        X = layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding='valid',
            activation='relu',
        )(input_layer)

        # max-pooling layer: pool_size = (3,3), strides = 2
        X = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

        # convolutional layer: filters = 64, strides = 1
        X = layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(X)

        # convolutional layer: filters = 192, kernel_size = (3,3)
        X = layers.Conv2D(
            filters=192, kernel_size=(3, 3), padding='same', activation='relu'
        )(X)

        # max-pooling layer: pool_size = (3,3), strides = 2
        X = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

        # 1st Inception block
        X = Inception_block(
            X,
            f1=64,
            f2_conv1=96,
            f2_conv3=128,
            f3_conv1=16,
            f3_conv5=32,
            f4=32,
        )

        # 2nd Inception block
        X = Inception_block(
            X,
            f1=128,
            f2_conv1=128,
            f2_conv3=192,
            f3_conv1=32,
            f3_conv5=96,
            f4=64,
        )

        # max-pooling layer: pool_size = (3,3), strides = 2
        X = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

        # 3rd Inception block
        X = Inception_block(
            X,
            f1=192,
            f2_conv1=96,
            f2_conv3=208,
            f3_conv1=16,
            f3_conv5=48,
            f4=64,
        )

        # Extra network 1:
        X1 = layers.AveragePooling2D(pool_size=(5, 5), strides=3)(X)
        X1 = layers.Conv2D(
            filters=128, kernel_size=(1, 1), padding='same', activation='relu'
        )(X1)
        X1 = layers.Flatten()(X1)
        X1 = layers.Dense(1024, activation='relu')(X1)
        X1 = layers.Dropout(0.7)(X1)
        X1 = layers.Dense(5, activation='softmax')(X1)

        # 4th Inception block
        X = Inception_block(
            X,
            f1=160,
            f2_conv1=112,
            f2_conv3=224,
            f3_conv1=24,
            f3_conv5=64,
            f4=64,
        )

        # 5th Inception block
        X = Inception_block(
            X,
            f1=128,
            f2_conv1=128,
            f2_conv3=256,
            f3_conv1=24,
            f3_conv5=64,
            f4=64,
        )

        # 6th Inception block
        X = Inception_block(
            X,
            f1=112,
            f2_conv1=144,
            f2_conv3=288,
            f3_conv1=32,
            f3_conv5=64,
            f4=64,
        )

        # Extra network 2:
        X2 = layers.AveragePooling2D(pool_size=(5, 5), strides=3)(X)
        X2 = layers.Conv2D(
            filters=128, kernel_size=(1, 1), padding='same', activation='relu'
        )(X2)
        X2 = layers.Flatten()(X2)
        X2 = layers.Dense(1024, activation='relu')(X2)
        X2 = layers.Dropout(0.7)(X2)
        X2 = layers.Dense(1000, activation='softmax')(X2)

        # 7th Inception block
        X = Inception_block(
            X,
            f1=256,
            f2_conv1=160,
            f2_conv3=320,
            f3_conv1=32,
            f3_conv5=128,
            f4=128,
        )

        # max-pooling layer: pool_size = (3,3), strides = 2
        X = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(X)

        # 8th Inception block
        X = Inception_block(
            X,
            f1=256,
            f2_conv1=160,
            f2_conv3=320,
            f3_conv1=32,
            f3_conv5=128,
            f4=128,
        )

        # 9th Inception block
        X = Inception_block(
            X,
            f1=384,
            f2_conv1=192,
            f2_conv3=384,
            f3_conv1=48,
            f3_conv5=128,
            f4=128,
        )

        # Global Average pooling layer
        X = layers.GlobalAveragePooling2D(name='GAPL')(X)

        # Dropoutlayer
        X = layers.Dropout(0.4)(X)

        # output layer
        X = layers.Dense(1000, activation='softmax')(X)

        # model
        return keras.Model(input_layer, [X, X1, X2], name='GoogLeNet')

    def __str__(self):
        super().__str__()
        self.get_model().summary(show_trainable=True)


def Inception_block(
    input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4
):
    # Input:
    # - f1: number of filters of the 1x1 convolutional layer in the first path
    # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
    # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
    # - f4: number of filters of the 1x1 convolutional layer in the fourth path

    # 1st path:
    path1 = layers.Conv2D(
        filters=f1, kernel_size=(1, 1), padding='same', activation='relu'
    )(input_layer)

    # 2nd path
    path2 = layers.Conv2D(
        filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu'
    )(input_layer)
    path2 = layers.Conv2D(
        filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu'
    )(path2)

    # 3rd path
    path3 = layers.Conv2D(
        filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu'
    )(input_layer)
    path3 = layers.Conv2D(
        filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu'
    )(path3)

    # 4th path
    path4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(
        input_layer
    )
    path4 = layers.Conv2D(
        filters=f4, kernel_size=(1, 1), padding='same', activation='relu'
    )(path4)
    output_layer = concatenate([path1, path2, path3, path4], axis=-1)
    # output_layer = layers.merge.concatenate([path1, path2, path3, path4], axis = -1)

    return output_layer
