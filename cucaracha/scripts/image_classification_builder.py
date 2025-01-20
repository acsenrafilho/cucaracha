import argparse
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from rich import print

from cucaracha.ml_models import image_classification
from cucaracha.ml_models.image_classification import *
from cucaracha.ml_trainers import ImageClassificationTrainer

# Script parameters
parser = argparse.ArgumentParser(
    prog='Image Classification Builder',
    description='Python script to build an Deep Learning image classification model.',
)
parser._action_groups.pop()
required = parser.add_argument_group(title='Required parameters')
optional = parser.add_argument_group(title='Optional parameters')


required.add_argument(
    '--dataset_path',
    type=str,
    required=True,
    help='Path to the dataset. This should follow the `cucaracha` dataset folder organization.',
)
required.add_argument(
    '--num_classes',
    type=int,
    required=True,
    help='The number of classes in the dataset. This must be defined based on the classes presented in the dataset.',
)
required.add_argument(
    '--img_shape',
    type=str,
    required=True,
    nargs='+',
    help='The image shape (height, width) to be used in the DL modeling. Pass it separeted by comma.',
)
required.add_argument(
    '--out_folder',
    type=str,
    required=True,
    help='The output folder where the model will be saved.',
)
optional.add_argument(
    '--verbose',
    action='store_true',
    help='Show more details thoughout the processing.',
)
optional.add_argument(
    '--arch',
    type=str,
    help='Define the model architecture to be used. If not provided, a default SmallXception architecture will be used. Remember that this must be a valid model architecture for image classification tasks. If you want to know all the available architectures, please run this script with the --arch_list flag.',
)
optional.add_argument(
    '--arch_list',
    action='store_true',
    help='List all the image classification architectures available in the Cucaracha framework.',
)
optional.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Define the batch size to be used in the training process. Default is 64. This must be a positive integer',
)
optional.add_argument(
    '--epochs',
    type=int,
    default=500,
    help='Define the number of epochs to be used in the training process. Default is 500. This must be a positive integer',
)
optional.add_argument(
    '--visualize_samples',
    action='store_true',
    help='If this is choosen then a training sample is genereated to be visualized before the training procedure to continure. The user will choose if the traninig can keep going or not.',
)

args = parser.parse_args()

# Execute --arch_list option and exit
if args.arch_list:
    print(' --- Image Classification Architectures Available ---')
    for arch in image_classification.__all__:
        print(f'Architecture: {arch}')
    print('Choose one of the above architectures to be used in the script.')
    exit()

# Script check-up parameters
def checkUpParameters():
    is_ok = True
    # Check output folder exist
    if not (os.path.isdir(args.out_folder)):
        print(
            f'Output folder path does not exist (path: {args.out_folder}). Please create the folder before executing the script.'
        )
        is_ok = False

    # Check dataset folder exist
    if not (os.path.isdir(args.dataset_path)):
        print(
            f'Dataset folder path does not exist (path: {args.dataset_path}).'
        )
        is_ok = False

    # Check image shape
    if len(img_shape) != 2:
        print(
            f'Image shape must be two values (height, width). Provided: {args.img_shape}'
        )
        is_ok = False

    return is_ok


try:
    img_shape = [int(s) for s in args.img_shape.split(',')]
except:
    img_shape = [int(s) for s in str(args.img_shape[0]).split(',')]

if not checkUpParameters():
    raise RuntimeError(
        'One or more arguments are not well defined. Please, revise the script call.'
    )

if args.verbose:
    print(' --- Script Input Data ---')
    print('Dataset path: ' + args.dataset_path)
    print('Image shape: ' + str(args.img_shape))
    print('Number of classes: ' + str(args.num_classes))
    print('Batch size: ' + str(args.batch_size))
    print('Epochs: ' + str(args.epochs))
    print('Output folder: ' + args.out_folder)

# Remove any folder called "logs" or "sample_visualization" in the output folder
for folder_name in ['logs', 'sample_visualization']:
    folder_path = os.path.join(args.out_folder, folder_name)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

# Step 1: Create the model architecture instance
model_architecture = SmallXception(
    img_shape=img_shape, num_classes=args.num_classes
)
if args.arch:
    model_architecture = eval(args.arch)(
        img_shape=img_shape, num_classes=args.num_classes
    )

# Step 2: Creathe the image classification trainer
trainer = ImageClassificationTrainer(
    dataset_path=args.dataset_path,
    num_classes=args.num_classes,
    architecture=model_architecture,
    img_shape=img_shape,
    batch_size=args.batch_size,
    epochs=args.epochs,
)

if args.verbose:
    print(' --- Image Classification Trainer ---')
    print(f'Epochs: {trainer.epochs}')
    print(f'Batch size: {trainer.batch_size}')
    print(f'Number of classes: {trainer.num_classes}')
    print(f'Optmizer: {trainer.optimizer}')
    print(f'Loss: {trainer.loss}')
    print(f'Metrics: {trainer.metrics}')
    print(f'Architecture name: {trainer.architecture.__class__.__name__}')

# #Step 3: Train the model
# Define a custom callback for TensorBoard image visualization
class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, dataset_path, img_shape):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.dataset_path = dataset_path
        self.img_shape = img_shape

    def on_epoch_end(self, epoch, logs=None):
        # Load a batch of images from the training set

        train_images = next(iter(trainer.dataset['train']))[0].numpy()
        train_images = (train_images * 255).astype(np.uint8)[0:30]
        train_images = train_images[1:]
        with self.file_writer.as_default():
            # Log the images to TensorBoard
            tf.summary.image(
                'Training images', train_images, step=epoch, max_outputs=30
            )


# Initialize the custom ImageLogger callback
image_logger = ImageLogger(
    log_dir=os.path.join(args.out_folder, 'logs', 'images'),
    dataset_path=args.dataset_path,
    img_shape=img_shape,
)

callback_list = [
    ModelCheckpoint(
        os.path.join(trainer.dataset_path, trainer.model_name),
        monitor='val_acc',
        save_best_only=True,
    ),
    EarlyStopping(monitor='val_loss', patience=10),
    TensorBoard(
        log_dir=os.path.join(args.out_folder, 'logs'), update_freq='batch'
    ),
    image_logger,
]

if args.visualize_samples:
    import matplotlib.pyplot as plt

    # Create the sample visualization folder
    sample_vis_folder = os.path.join(args.out_folder, 'sample_visualization')
    os.makedirs(sample_vis_folder, exist_ok=True)

    # Load a batch of images from the training set
    train_images = trainer.collect_training_samples(200)
    train_images = train_images.astype(np.uint8)

    # Save the images to the sample visualization folder
    for i in range(len(train_images)):
        plt.imsave(
            os.path.join(sample_vis_folder, f'sample_{i}.png'), train_images[i]
        )

    print(f'Sample images saved to {sample_vis_folder}')
    user_input = input(
        '[bold green]Do you want to continue the training? (yes/no): '
    )
    if user_input.lower() != 'yes':
        print('Training stopped by the user.')
        exit()


trainer.train_model(callbacks=callback_list)

# Finish
model_save_full_path = os.path.join(args.out_folder, trainer.model_name)
trainer.model.save(model_save_full_path)
# Save the class labels as a JSON file

model_name_without_extension = os.path.splitext(trainer.model_name)[0]
class_labels_path = os.path.join(
    args.out_folder, f'{model_name_without_extension}_class_labels.json'
)
with open(class_labels_path, 'w') as f:
    json.dump(trainer.class_names, f)

if args.verbose:
    print(f'Class labels saved successfully in: {class_labels_path}')
if args.verbose:
    print(
        f'Model training completed successfully and saved in: {args.out_folder}{trainer.model_name}.'
    )
print(' --- Script Execution Completed ---')
