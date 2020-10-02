import numpy as np
import idx2numpy
import math
import random

training_image_file = "./data/train-images-idx3-ubyte"
training_label_file = "./data/train-labels-idx1-ubyte"

images = idx2numpy.convert_from_file(training_image_file)
dataset_array = images.tolist()


def get_random_duplicates(dataset, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(1, len(dataset)), duplicate_amount)
    result = dataset
    for i in random_indexes:
        result.append(dataset[i])

    return result


print(len(get_random_duplicates(dataset_array, 10)))

