import keras

from cucaracha.ml_models.model_architect import ModelArchitect


class ResNet50(ModelArchitect):
    """
    ResNet50 is a custom model architecture for image classification tasks,
    inheriting from the ModelArchitect base class. This model is based on the
    ResNet50 architecture, designed to handle large-scale image classification
    tasks with high accuracy and efficiency.

    Reference:
        He, K., Zhang, X., Ren, S., & Sun, J. (2016).
        Deep residual learning for image recognition.
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.

    Attributes:
        img_shape (tuple): The shape of the input images (height, width).
        num_classes (int): The number of output classes for classification.

    Methods:
        get_model():
            Builds and returns the Keras model based on the ResNet50 architecture.
        __str__():
            Returns a string representation of the model, including a summary of the
            model architecture with trainable parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(modality='image_classification', **kwargs)
        self.img_shape = kwargs.get('img_shape')
        self.num_classes = kwargs.get('num_classes')

    def get_model(self):
        return keras.applications.resnet50.ResNet50(
            weights=None,
            input_shape=(self.img_shape[0], self.img_shape[1], 3),
            classes=self.num_classes,
        )

    def __str__(self):
        output = super().__str__()
        self.get_model().summary(show_trainable=True)
        return output
