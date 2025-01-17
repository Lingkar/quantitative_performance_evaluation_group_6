import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import clone_model
from datasets import get_data, get_training_data
from models import get_model
from util import inject_noise, select_clean_noisy, combine_result, BNN_active_selection, y_weak_generator
from util import model_reliability, BNN_label_predict, label_decider, stat_func
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist
import time
import argparse
from tensorflow.python.lib.io import file_io
from keras.utils import np_utils, multi_gpu_model
from keras import backend as K
from io import BytesIO
import os
import pickle
import pdb
import time
import idx2numpy
import math
import random

np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
start = time.time()
NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'celeb': 20}
dataset = "mnist"


def train(dataset, alpha, beta, thr, ks1, ks2, epochs, error_ratio, image_noise, labeler, replication_set):
    # Boolean to enable or disable labeler:
    if int(labeler) == 0:
        labeler = False
    else:
        labeler = True

    # Here we import all the datasets once
    X_train_orig = idx2numpy.convert_from_file(
        "../data/experiments/repetition_%d/le_%d_in_%d/train-images-idx3-ubyte" % (replication_set, error_ratio, image_noise))
    y_train_orig = idx2numpy.convert_from_file(
        "../data/experiments/repetition_%d/le_%d_in_%d/train-labels-idx1-ubyte" % (replication_set, error_ratio, image_noise))
    # These two can be taken from the arbitrary files as these are not changed.
    X_test_orig = idx2numpy.convert_from_file(
        "../data/t10k-images-idx3-ubyte")
    y_test_orig = idx2numpy.convert_from_file(
        "../data/t10k-labels-idx1-ubyte")
    
    # These are the clean datasets (Used for getting correct label and for validation set (maybe weird???))
    X_train_clean = idx2numpy.convert_from_file(
        "../data/train-images-idx3-ubyte"
    )
    y_train_clean = idx2numpy.convert_from_file(
        "../data/train-labels-idx1-ubyte"
    )

    alpha = alpha
    beta = beta
    thr = thr
    ks1 = ks1
    ks2 = ks2
    image_noise = image_noise
    w_lim = 1
    # The cost of the strong labeler
    c = 2
    # would cost only to use weak labeler
    batch_budget = 1000
    bnn = 0
    al_method = 'bvssb'
    dynamic = 0
    budget = 0
    epochs_init = 5
    batch_size = 128
    epochs = epochs
    dataset = dataset
    init_noise_ratio = 0
    data_ratio = 1.666666
    error_ratio = error_ratio

    # This is just for the model fit init, and to set the validation set, in the homework the init_noise_ratio is set 0
    X_train, y_train, X_test, y_test, x_val, y_val, un_selected_index = get_data(X_train_clean, y_train_clean,
                                                                                 X_test_orig, y_test_orig, data_ratio)
    # y_val and x_val shape changes based on the data_ratio used. Not sure yet what should be used.
    # print(y_val.shape)
    print("train shape for the pre-trained model:", X_train.shape)
    print("length of un_selected_index:", len(un_selected_index))

    image_shape = X_train.shape[1:]
    model = get_model(bnn, dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset],
                      ks1=ks1, ks2=ks2)
    optimizer = SGD(lr=0.01, decay=0.0002, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    datagen.fit(X_train)

    h_quality = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs_init,
                                    validation_data=(X_test, y_test)
                                    )
    model_quality = clone_model(model)
    model_quality.set_weights(model.get_weights())

    # Set the train test sets to the original sets (with noise and label errors)
    X_train = X_train_orig
    X_test = X_test_orig
    y_train = y_train_orig
    y_test = y_test_orig
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    means = X_train.mean(axis=0)
    # std = np.std(X_train)
    X_train = (X_train - means)  # / std
    X_test = (X_test - means)  # / std
    # they are 2D originally in cifar

    # Do exactly the same operation for the y_train clean as for the dirty y_train
    y_train = y_train.ravel()
    y_train_clean = y_train_clean.ravel()
    y_test = y_test.ravel()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_train_clean = np_utils.to_categorical(y_train_clean, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])
    epochs_training = epochs
    print("y_train clean:", y_train_clean.shape)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)
    n_classes = NUM_CLASSES[dataset]
    # training_steps indicate how many data used in each batch
    training_steps = 1000
    statistics_list = [[] for _ in range(21)]
    steps = int(np.ceil(len(un_selected_index) / float(training_steps)))

    for i in np.arange(steps):
        print("chunk:", i)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        if i == 0:
            sub_un_selected_list = un_selected_index[0:training_steps]
        elif i != steps - 1:
            sub_un_selected_list = un_selected_index[i * training_steps:(i + 1) * training_steps]
        else:
            sub_un_selected_list = un_selected_index[i * training_steps:]
        if i != 0:
            model_quality = clone_model(model)
            model_quality.set_weights(model.get_weights())

        X_in_iteration = X_train[sub_un_selected_list]
        # Set the y_true_iteretion to the subset of our y_train_clean data-set
        y_true_iteration = y_train_clean[sub_un_selected_list]
        y_noisy_iteration = y_train[sub_un_selected_list]

        # This line is not used anymore as we have introduced the label-errors our selves in our datasets
        # y_noisy_iteration, noisy_idx = introduce_label_errors(y_train[sub_un_selected_list], error_ratio)

        y_predict, predict_prob, y_predict_label_second = BNN_label_predict(X_in_iteration, model, n_classes)
        clean_list, clean_pred_list, noisy_list, FN, TN, TP, FP, maxProbTP, maxProbFP = select_clean_noisy(
            X_in_iteration,
            y_noisy_iteration, y_true_iteration, y_predict, y_predict_label_second, predict_prob, model)

        if batch_budget > 0:
            num_al = len(noisy_list)
        print("batch_budget:", batch_budget)

        batch_list = range(0, training_steps)
        n = NUM_CLASSES[dataset]
        spent_s = 0
        spent_w = 0

        # The labeler is needed in this class as here the active selection for the most incorrect labels is made
        inf_ind, inf = BNN_active_selection(predict_prob, noisy_list, al_method, num_al, labeler)
        loss = model_reliability(model, x_val, y_val)

        # reserve_list is also added in util.py
        reserve_list, strong_list, weak_list, spent_s, spent_w, first_yes, weak_questions, mean_w_q, unique_pic, true_label_rank = label_decider(
            noisy_list, y_true_iteration, y_predict, predict_prob, n, inf_ind, inf, loss, batch_budget, spent_s,
            spent_w, w_lim, c, alpha, beta, thr)

        statistics_list[0].append(len(strong_list))
        statistics_list[1].append(len(weak_list))
        statistics_list[2].append(first_yes)
        statistics_list[3].append(spent_w)
        statistics_list[4].append(mean_w_q)
        statistics_list[5].append(unique_pic)
        statistics_list[6].append(num_al)

        print("spent_s:", spent_s, "spent_s/c:", spent_s / c)
        print("spent_w:", spent_w, "weak yes:", len(weak_list))

        oracle_list = np.append(strong_list, weak_list)
        if len(weak_list) == 0:
            oracle_list = strong_list
        if len(strong_list) == 0:
            oracle_list = weak_list

        # Don't do this line in order to stop the labeler
        if labeler:
            y_noisy_iteration[oracle_list] = y_true_iteration[oracle_list]

        # Add the reserve_list to not reduce the training size
        # training_list = np.append(clean_list, oracle_list) # This line is not used anymore
        training_list = []
        training_list.extend(clean_list)
        training_list.extend(oracle_list)
        # training_list.extend(reserve_list)
        print(len(training_list))

        # I am not totally sure, but this selects the data it wants to train on don't do that if labeler false
        if labeler:
            x_train_iteration = X_in_iteration[training_list]
            y_train_iteration = y_noisy_iteration[training_list]

        else:
            x_train_iteration = X_in_iteration
            y_train_iteration = y_noisy_iteration

        print("train data shape:", x_train_iteration.shape)
        h_training_epoch_quality = model.fit_generator(
            datagen.flow(x_train_iteration, y_train_iteration, batch_size=batch_size),
            steps_per_epoch=y_train_iteration.shape[0] // batch_size, epochs=epochs_training,
            validation_data=(X_test, y_test)
        )
        h_quality = combine_result(h_quality, h_training_epoch_quality)
        if i != 0 and h_quality.history['val_accuracy'][-epochs_training - 1] - np.min(
                h_quality.history['val_accuracy'][-epochs_training:]) > 0.2:
            model = clone_model(model_quality)
            model.set_weights(model_quality.get_weights())

    s_bnn = 0
    val_accc = h_quality.history['val_accuracy']
    accc = h_quality.history['accuracy']
    val_losss = h_quality.history['val_loss']
    losss = h_quality.history['loss']
    for i in range(0, 10):
        s_bnn = s_bnn + val_accc[-epochs_training * i - 1]
        print(val_accc[-epochs_training * i - 1])

    average_valacc = s_bnn / 10
    print("Final Acc:", average_valacc)
    statistics_list[14].append(val_accc)
    statistics_list[15].append(accc)
    statistics_list[16].append(val_losss)
    statistics_list[17].append(losss)
    end = time.time()
    statistics_list[18].append(end - start)
    print("timing: ", end - start)
    #################################### save ########################################
    with open('WSstat_alpha' + str(alpha) + '+beta' + str(beta) + '+threshold' + str(thr) + '+ks1' + str(
            ks1) + '+ks2' + str(ks2) + '+epochs' + str(epochs) + '+error_ration' + str(
            error_ratio) + '+image_noise' + str(image_noise) + '+labeler' + str(labeler) + '+replication_set' + str(replication_set) + '.pickle', 'wb') as file_pi:
        pickle.dump(statistics_list, file_pi)
    ##################################################################################
    print(average_valacc)
    return statistics_list


def main(args):
    assert args.dataset in ['mnist', 'cifar-10', 'cifar-100'], \
        "dataset parameter must be either 'mnist', 'cifar-10', 'cifar-100'"
    train(args.dataset, args.alpha, args.beta, args.thr, args.ks1, args.ks2, args.epochs, args.error_ratio,
          args.image_noise, args.labeler, args.replication_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar-10', 'cifar-100'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--alpha',
        help="I value's weight",
        required=True, type=float
    )
    parser.add_argument(
        '-b', '--beta',
        help="cost avlue's weight",
        required=True, type=float
    )
    parser.add_argument(
        '-t', '--thr',
        help="Q function's threshold",
        required=True, type=float
    )
    parser.add_argument(
        '-ks1', '--ks1',
        help="kernel size of convolution1",
        required=True, type=int
    )
    parser.add_argument(
        '-ks2', '--ks2',
        help="kernel size of convolution2",
        required=True, type=int
    )
    parser.add_argument(
        '-e', '--epochs',
        help="training epochs",
        required=True, type=int
    )
    parser.add_argument(
        '-er', '--error_ratio',
        help="ratio of label errors",
        required=True, type=int
    )
    parser.add_argument(
        '-in', '--image_noise',
        help="ratio of image noise",
        required=True, type=int
    )
    parser.add_argument(
        '-la', '--labeler',
        help="Set 0 or 1 in order to use labeler.",
        required=True, type=int
    )
    parser.add_argument(
        '-re', '--replication_set',
        help="Set to the replication set which you want to use",
        required=True, type=int
    )
    args = parser.parse_args()
    main(args)


