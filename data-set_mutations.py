import numpy as np
import idx2numpy
import math
import random
import gzip

training_image_file = "./data/train-images-idx3-ubyte"
training_label_file = "./data/train-labels-idx1-ubyte"

images = idx2numpy.convert_from_file(training_image_file)
labels = idx2numpy.convert_from_file(training_label_file)


def add_random_duplicates(dataset, labels, percentage):
    dataset_list = dataset.tolist()
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(0, len(dataset)), duplicate_amount)
    dup_values = [dataset[i] for i in random_indexes]
    dup_labels = [labels[i] for i in random_indexes]

    result = np.append(dataset, dup_values, axis=0)
    labels = np.append(labels, dup_labels, axis=0)

    return result, labels


def delete_random_samples(dataset, labels, percentage):
    duplicate_amount = math.floor((len(dataset) * percentage / 100))
    random_indexes = random.sample(range(0, len(dataset)), duplicate_amount)

    result = [i for j, i in enumerate(dataset) if j not in random_indexes]
    labels = [i for j, i in enumerate(labels) if j not in random_indexes]

    return np.array(result), np.array(labels)


def introduce_label_errors(input_labels, percentage):
    labels = np.array(input_labels, copy=True)
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


def shuffle_data(dataset, labels):
    random_indexes = random.sample(range(0, len(dataset)), len(dataset))
    result_dataset = [dataset[i] for i in random_indexes]
    result_labels = [labels[i] for i in random_indexes]

    return np.array(result_dataset), np.array(result_labels)


def sp_noise(image, percentage):
    output = np.zeros(image.shape, np.uint8)
    treshold = 1 - (percentage/100)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < (percentage/100):
                output[i][j] = 0
            elif rdn > treshold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_noise(dataset, dataset_percentage, noise_percentage):
    result = np.array(dataset, copy=True)
    image_amount = math.floor((len(dataset) * dataset_percentage / 100))
    random_indexes = random.sample(range(0, len(dataset)), image_amount)

    for i in random_indexes:
        result[i] = sp_noise(dataset[i], noise_percentage)

    return result


# add 10% random duplicates
dup_result, dup_labels = add_random_duplicates(images, labels, 10)
dup_filenames = [
    "./data/random_duplicates/train-images-idx3-ubyte",
    "./data/random_duplicates/train-labels-idx1-ubyte"
]
idx2numpy.convert_to_file(dup_filenames[0], dup_result)
idx2numpy.convert_to_file(dup_filenames[1], dup_labels)

for filename in dup_filenames:
    with open(filename, 'rb') as f_in:
        with gzip.open('%s.gz'%filename, 'wb') as f_out:
            f_out.writelines(f_in)


# delete 10% random samples
shrinked_result, shrinked_labels = delete_random_samples(images, labels, 10)
shrinked_filenames = [
    "./data/delete_random_samples/train-images-idx3-ubyte",
    "./data/delete_random_samples/train-labels-idx1-ubyte"
]

idx2numpy.convert_to_file(shrinked_filenames[0], shrinked_result)
idx2numpy.convert_to_file(shrinked_filenames[1], shrinked_labels)

for filename in shrinked_filenames:
    with open(filename, 'rb') as f_in:
        with gzip.open('%s.gz'%filename, 'wb') as f_out:
            f_out.writelines(f_in)


# Introduce 10% label errors
erroneous_labels = introduce_label_errors(labels, 10)
erroneous_labels_name = "./data/label_errors/train-labels-idx1-ubyte"

idx2numpy.convert_to_file(erroneous_labels_name, erroneous_labels)

with open(erroneous_labels_name, 'rb') as f_in:
    with gzip.open('%s.gz'%erroneous_labels_name, 'wb') as f_out:
        f_out.writelines(f_in)


shuffled_result, shuffled_labels = shuffle_data(images, labels)
shuffled_filenames = [
    "./data/shuffled/train-images-idx3-ubyte",
    "./data/shuffled/train-labels-idx1-ubyte"
]

idx2numpy.convert_to_file(shuffled_filenames[0], shuffled_result)
idx2numpy.convert_to_file(shuffled_filenames[1], shuffled_labels)

for filename in shuffled_filenames:
    with open(filename, 'rb') as f_in:
        with gzip.open('%s.gz'%filename, 'wb') as f_out:
            f_out.writelines(f_in)


noise_result = add_noise(images, 20, 2)
noise_filenames = [
    "./data/noise/train-images-idx3-ubyte",
    "./data/noise/train-labels-idx1-ubyte"
]

idx2numpy.convert_to_file(noise_filenames[0], noise_result)
idx2numpy.convert_to_file(noise_filenames[1], labels)

for filename in noise_filenames:
    with open(filename, 'rb') as f_in:
        with gzip.open('%s.gz'%filename, 'wb') as f_out:
            f_out.writelines(f_in)