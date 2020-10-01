import numpy as np
import idx2numpy
import math
import random

from numpy.core._multiarray_umath import ndarray

training_image_file = "./data/train-images-idx3-ubyte"
training_label_file = "./data/train-labels-idx1-ubyte"

images = idx2numpy.convert_from_file(training_image_file)

print(images.shape)


def get_random_duplicates(dataset, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(1, len(dataset)), duplicate_amount)
    duplicated_images = []
    for i in random_indexes:
        duplicated_images.append(dataset[i])

    duplicated_images_np = np.asarray(duplicated_images)
    new_dataset: ndarray = np.append(dataset, duplicated_images_np)
    return new_dataset


print(get_random_duplicates(images, 10).shape)
