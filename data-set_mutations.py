import numpy as np
import idx2numpy
import math
import random

training_image_file = "./data/train-images-idx3-ubyte"
training_label_file = "./data/train-labels-idx1-ubyte"

label = idx2numpy.convert_from_file(training_label_file)[0:10]
images = idx2numpy.convert_from_file(training_image_file)

label_array = label.tolist()
dataset_array = images.tolist()


def add_random_duplicates(dataset, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(0, len(dataset)), duplicate_amount)
    result = dataset
    for i in random_indexes:
        result.append(dataset[i])

    return result


def delete_random_samples(dataset, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(0, len(dataset)), duplicate_amount)

    result = [i for j, i in enumerate(dataset) if j not in random_indexes]

    return result


def introduce_label_errors(labels, percentage):
    max_val = max(labels)
    min_val = min(labels)

    duplicate_amount = math.floor((len(labels) * percentage / 100))

    random_indexes = random.sample(range(0, len(labels)), duplicate_amount)

    for i in random_indexes:
        dirty_label = random.sample(range(min_val, max_val + 1), 1)[0]
        while dirty_label == labels[i]:
            dirty_label = random.sample(range(min_val, max_val + 1), 1)[0]

        labels[i] = dirty_label

    return labels

