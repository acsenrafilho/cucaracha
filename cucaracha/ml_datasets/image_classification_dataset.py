import math

import cv2
import numpy as np
import tensorflow as tf


class ImageClassificationDataset(tf.keras.utils.PyDataset):
    def __init__(self, x_set, y_set, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.image_size = (200, 200)

    def __len__(self):
        # Return number of batches.
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # Return x, y for batch idx.
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        batch_dataset = []

        for file_name in batch_x:
            img = cv2.resize(cv2.imread(file_name), self.image_size)
            batch_dataset.append(img)

        # return np.array([
        #     resize(cv2.imread(file_name), self.image_size)
        #        for file_name in batch_x]), np.array(batch_y)
        return np.array(batch_dataset), np.array(batch_y)

    # np.array([resize(cv2.imread(file_name), self.image_size) for file_name in batch_x]), np.array(batch_y)
