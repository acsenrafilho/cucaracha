import keras
from keras import layers

from cucaracha.ml_models.model_architect import ModelArchitect


class SmallXception(ModelArchitect):
    def __init__(self, **kwargs):
        super().__init__(modality='image_classification', **kwargs)
        self.img_shape = kwargs.get('img_shape')
        self.num_classes = kwargs.get('num_classes')

    def get_model(self):
        input_shape = (self.img_shape[0], self.img_shape[1], 3)
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        previous_block_activation = x  # Set aside residual

        for size in [256, 512, 728]:
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(size, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(size, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding='same')(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling2D()(x)
        if self.num_classes == 2:
            units = 1
        else:
            units = self.num_classes

        x = layers.Dropout(0.25)(x)
        # We specify activation=None so as to return logits
        outputs = layers.Dense(units, activation='softmax')(x)

        return keras.Model(inputs, outputs)

    def __str__(self):
        super().__str__()
        self.get_model().summary(show_trainable=True)
