import numpy as np
import idx2numpy
import math
import random

training_image_file = "./data/train-images-idx3-ubyte"
training_label_file = "./data/train-labels-idx1-ubyte"

images = idx2numpy.convert_from_file(training_image_file)
dataset_array = images.tolist()


def add_random_duplicates(dataset, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(1, len(dataset)), duplicate_amount)
    result = dataset
    for i in random_indexes:
        result.append(dataset[i])

    return result


def delete_random_samples(dataset, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(1, len(dataset)), duplicate_amount)
    count = 0

    result = [i for j, i in enumerate(dataset) if j not in random_indexes]

    return result



print(len(delete_random_samples(dataset_array, 34)))
