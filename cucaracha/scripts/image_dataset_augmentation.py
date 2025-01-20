# Make a local data augmentation to create a new dataset
# Add augmentation using "alpha" expansion of images (other as backgrounda nd class as foreground)
# Save locally the augmented dataset
import argparse
import os
import shutil
import warnings

import cv2 as cv
import numpy as np
from keras import layers
from rich import print

from cucaracha.configuration import ALLOWED_IMAGE_EXTENSIONS
from cucaracha.utils import verify_image_compatibility

# Script parameters
parser = argparse.ArgumentParser(
    prog='Image Dataset Agumentation',
    description='Python script to create a new dataset based on data augmentation.',
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
    '--img_shape',
    type=str,
    nargs='+',
    help='The image shape (height, width) to be used in the DL modeling. Pass it separeted by comma. If not provided, the original image shape will be used. If not provided, the shape (256,256) is used.',
)
optional.add_argument(
    '--bg_sample',
    type=str,
    help='Define the background sample to be used in the data augmentation process. This should be a path to a folder containing images to be used as background. It is important that all the background images is placed into a single folder. If not provided, the data augmentation process will be done without background images.',
)
optional.add_argument(
    '--batch_size',
    type=int,
    default=20,
    help='Define the batch size to be load data from the input data path. This is useful to prevent memory flooding from the reading image process. If not provided, the default value is 20. For each batch, the same random bakcround image is applied (is choosen).',
)

args = parser.parse_args()


# TODO Colocar check parameters para os inputs do scripts


def _augment_batch_and_save(batch):
    # TODO Make it saving the image using the same data structure (same classes)
    augmented_images = []

    counter = 0
    for img, bg in batch:
        counter += 1
        augmented_img = augmenter(img)
        try:
            if bg is None:
                augmented_images.append(augmented_img)
                continue

            combined_img = np.copy(bg)
            mask = ~np.isnan(augmented_img)
            combined_img[mask] = augmented_img[mask]

            # augmented_img = augmenter(np.expand_dims(combined_img, axis=0))
            augmented_images.append(combined_img)
        except Exception as e:
            warnings.warn(
                f'Problem to create the {counter} augmented image: {e} /nSkipped this image and continue...'
            )
            continue

    return augmented_images


fill_value = 0
if args.bg_sample:
    fill_value = np.NaN

data_aug = [
    layers.RandomFlip(),
    layers.RandomRotation(0.3, fill_mode='constant', fill_value=fill_value),
    layers.RandomZoom(
        (-0.2, 0.5), fill_mode='constant', fill_value=fill_value
    ),
    layers.RandomShear(0.3, fill_mode='constant', fill_value=fill_value),
    layers.RandomTranslation(
        (-0.2, 0.2), 0.1, fill_mode='constant', fill_value=fill_value
    ),
]


def augmenter(images):
    for op in data_aug:
        images = op(images)

    return images


input_collector = os.walk(args.dataset_path)
if args.bg_sample:
    bg_collector = os.walk(args.bg_sample)
    bg_list = []
    for root, _, files in bg_collector:
        for file in files:
            if file.endswith(ALLOWED_IMAGE_EXTENSIONS):
                bg_list.append(file)


# counter_aug = 0
batch_size = args.batch_size
img_shape = (
    tuple(map(int, args.img_shape[0].split(',')))
    if args.img_shape
    else ['256,256']
)


if os.path.exists(args.out_folder):
    shutil.rmtree(args.out_folder)
os.makedirs(args.out_folder)


# for bg_root, _, bg_files in bg_collector:
def _collect_bg_image():
    rand_bg_file = np.random.choice(bg_list)
    bg_path = os.path.join(root, rand_bg_file)
    bg_img = cv.resize(cv.imread(bg_path), img_shape)

    return bg_img


bg_img = None
for in_root, _, in_files in input_collector:
    if args.bg_sample:
        bg_img = _collect_bg_image()

    batch_counter = 0
    aug_counter = 0
    batch = []
    for img_file in in_files:
        if img_file.endswith(ALLOWED_IMAGE_EXTENSIONS):
            img_path = os.path.join(in_root, img_file)
            img = cv.resize(cv.imread(img_path), img_shape)
            batch.append((img, bg_img))

            if len(batch) % batch_size == 0:
                batch_counter += 1
                if args.bg_sample:
                    bg_img = _collect_bg_image()
                aug_batch = _augment_batch_and_save(batch)
                # TODO save image files with same input data structure (same classes)

                if args.verbose:
                    print(
                        f'[bold green]Batch {batch_counter}: Generated {len(batch)} augmented images.'
                    )
                for i, img in enumerate(aug_batch):
                    output_path = os.path.join(
                        args.out_folder,
                        f'batch_{batch_counter}_augmented_{i}.jpg',
                    )
                    aug_counter += 1
                    cv.imwrite(output_path, img)

                # Save the new dataset locally
                if args.verbose:
                    print(
                        f'[bold green]Batch {batch_counter}: Saved augmented images to {args.out_folder}.'
                    )


# Save the new dataset locally
if args.verbose:
    print(
        f'[bold green]All augmentation is done! Total of {aug_counter} images saved at: {args.out_folder}.'
    )
