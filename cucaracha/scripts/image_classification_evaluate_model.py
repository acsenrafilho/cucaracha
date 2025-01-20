import argparse
import json
import os
import shutil

import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from rich import print
from sklearn.metrics import classification_report, confusion_matrix

# Script parameters
parser = argparse.ArgumentParser(
    prog='Image Classification for model evaluation',
    description='Python script to evaluate a pretrained Deep Learning image classification model.',
)
parser._action_groups.pop()
required = parser.add_argument_group(title='Required parameters')
optional = parser.add_argument_group(title='Optional parameters')


required.add_argument(
    '--evaluate_dataset',
    type=str,
    required=True,
    help='Path to the evaluate dataset. This should follow the `cucaracha` dataset folder organization and should be a folder containing the images to be evaluated.',
)
required.add_argument(
    '--model',
    type=str,
    required=True,
    help='Path to the model to be evaluated. This must be an image classification model that can be loaded from Keras framework.',
)
required.add_argument(
    '--labels',
    type=str,
    required=True,
    help='Path to the class labels JSON file that is generated from the cucaracha image classification builder script.',
)
optional.add_argument(
    '--verbose',
    action='store_true',
    help='Show more details thoughout the processing.',
)
optional.add_argument(
    '--save_evaluation',
    action='store_true',
    help='Determines if the evaluation results will be saved in the evaluate dataset folder. If this flag is not provided, the evaluation results will not be saved.',
)

args = parser.parse_args()


def _create_grid_sample(labels, X_test, y_pred=None):
    # Let's view more images in a grid format
    # Define the dimensions of the plot grid
    W_grid = 5
    L_grid = 5
    subsample_size = W_grid * L_grid  # Define the size of the subsample

    fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
    axes = axes.ravel()   # flaten the 15 x 15 matrix into 225 array

    # Convert a subsample of evaluate_dataset to a numpy array
    subsample = evaluate_dataset.take(subsample_size)

    X_test = []
    y_true = []
    for images, labels in subsample:
        X_test.append(images.numpy())
        y_true.extend(labels.numpy())

    X_test = np.concatenate(X_test, axis=0)
    # y_true = np.concatenate(y_true, axis=0)

    n_test = subsample_size   # get the length of the train dataset

    # Select a random number from 0 to n_train
    for i in np.arange(0, subsample_size):   # create evenly spaces variables

        # Select a random number
        index = np.random.randint(0, n_test)
        # read and display an image with the selected index
        axes[i].imshow(X_test[index].astype(np.uint8))
        if y_pred is not None:
            label_index = int(y_pred[index])
            axes[i].set_title(str(int(labels[label_index])), fontsize=8)
        else:
            label_index = int(y_true[index])
            axes[i].set_title(str(int(labels[label_index])), fontsize=8)
        axes[i].axis('off')

    plt.subplots_adjust(hspace=0.4)

    return plt


# Script check-up parameters
def checkUpParameters():
    is_ok = True
    # Check dataset folder exist
    if not (os.path.isdir(args.evaluate_dataset)):
        print(
            f'Dataset folder path does not exist (path: {args.evaluate_dataset}).'
        )
        is_ok = False

    # Check model file exist
    if not (os.path.isfile(args.model)):
        print(f'Model file does not exist (path: {args.model}).')
        is_ok = False

    return is_ok


# Check-up parameters
if not checkUpParameters():
    raise RuntimeError(
        'One or more arguments are not well defined. Please, revise the script call.'
    )

# Remove evaluate folder if it exists
evaluation_folder = os.path.join(args.evaluate_dataset, 'evaluation')
if os.path.exists(evaluation_folder):
    if os.path.isdir(evaluation_folder):
        shutil.rmtree(evaluation_folder)


# Load the model input
model = keras.models.load_model(args.model)

# Load the evaluate dataset
evaluate_dataset = keras.utils.image_dataset_from_directory(
    directory=args.evaluate_dataset,
    image_size=model.input_shape[1:3],
    verbose=args.verbose,
)

class_names_json = {}
try:
    with open(args.labels, 'r') as f:
        class_names_json = json.load(f)
except:
    raise ValueError(f'Error loading class labels from file: {args.labels}')
num_classes = len(class_names_json)


y_pred = []
y_true = []
for images, labels in evaluate_dataset:
    y_true.extend(
        [
            list(class_names_json.values()).index(
                evaluate_dataset.class_names[label]
            )
            for label in labels.numpy()
        ]
    )
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=-1))
y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Classification report
print('Classification Report:')
report = classification_report(
    y_true, y_pred, target_names=list(class_names_json.values())
)
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=list(class_names_json.values()),
    yticklabels=list(class_names_json.values()),
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

os.makedirs(evaluation_folder, exist_ok=True)
# Save confusion matrix
conf_matrix_path = os.path.join(evaluation_folder, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
print(f'Confusion matrix saved at: {conf_matrix_path}')

# Save classification report
classification_report_path = os.path.join(
    evaluation_folder, 'classification_report.txt'
)
with open(classification_report_path, 'w') as f:
    f.write(report)
print(f'Classification report saved at: {classification_report_path}')

# Save a grid figure showing some predictions
grid_plt = _create_grid_sample(
    list(class_names_json.values()), evaluate_dataset
)
grid_plt_path = os.path.join(evaluation_folder, 'grid_input_sample.png')
grid_plt.savefig(grid_plt_path)

y_pred_evaluate = np.argmax(model.predict(evaluate_dataset), axis=-1)
grid_plt = _create_grid_sample(
    list(class_names_json.values()), evaluate_dataset, y_pred_evaluate
)
grid_plt_path = os.path.join(evaluation_folder, 'grid_prediction_sample.png')
grid_plt.savefig(grid_plt_path)
