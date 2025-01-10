import argparse
import json
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from rich import print

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
    help='Path to the dataset. This should follow the `cucaracha` dataset folder organization.',
)
required.add_argument(
    '--num_classes',
    type=int,
    help='The number of classes in the dataset. This must be defined based on the classes presented in the dataset.',
)
required.add_argument(
    '--img_shape',
    type=str,
    nargs='+',
    help='The image shape (height, width) to be used in the DL modeling. Pass it separeted by comma.',
)
required.add_argument(
    '--out_folder',
    type=str,
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
    help='Define the model architecture to be used. If not provided, a default SmallXception architecture will be used. Remember that this must be a valid model architecture for image classification tasks.',
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

args = parser.parse_args()

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
    print(f'Optmizer: {trainer.optmizer}')
    print(f'Loss: {trainer.loss}')
    print(f'Metrics: {trainer.metrics}')
    print(f'Architecture name: {trainer.architecture.__class__.__name__}')

# #Step 3: Train the model
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
]
trainer.train_model(callbacks=callback_list)

# Finish
model_save_full_path = os.path.join(args.out_folder, trainer.model_name)
trainer.model.save(model_save_full_path)
# Save the class labels as a JSON file

class_labels_path = os.path.join(args.out_folder, 'class_labels.json')
with open(class_labels_path, 'w') as f:
    json.dump(trainer.class_names, f)

if args.verbose:
    print(f'Class labels saved successfully in: {class_labels_path}')
if args.verbose:
    print(
        f'Model training completed successfully and saved in: {args.out_folder}{trainer.model_name}.'
    )
