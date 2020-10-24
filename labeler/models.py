"""
#Trains a ResNet on the CIFAR10 dataset.
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
ResNet v2:
[Identity Mappings in Deep Residual Networks
](https://arxiv.org/pdf/1603.05027.pdf)
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v1|  3| 92.16 %|                 91.25 %|35
ResNet32   v1|  5| 92.46 %|                 92.49 %|50
ResNet44   v1|  7| 92.50 %|                 92.83 %|70
ResNet56   v1|  9| 92.71 %|                 93.03 %|90
ResNet110  v1| 18| 92.65 %|            93.39+-.16 %|165
ResNet164  v1| 27|     - %|                 94.07 %|  -
ResNet1001 v1|N/A|     - %|                 92.39 %|  -
&nbsp;
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v2|  2|     - %|                     - %|---
ResNet32   v2|N/A| NA    %|            NA         %| NA
ResNet44   v2|N/A| NA    %|            NA         %| NA
ResNet56   v2|  6| 93.01 %|            NA         %|100
ResNet110  v2| 12| 93.15 %|            93.63      %|180
ResNet164  v2| 18|     - %|            94.54      %|  -
ResNet1001 v2|111|     - %|            95.08+-.14 %|  -
"""

import keras
import pdb
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout
from keras.layers import AveragePooling2D, Input, Flatten

def get_model(bnn, dataset='mnist', input_tensor=None, input_shape=None, num_classes=10, ks1=3, ks2=3):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar-10' or 'cifar-100') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    input_shape: optional shape tuple
    :return: The model; a Keras 'Model' instance.
    """
    assert dataset in ['mnist','fashion_mnist', 'svhn', 'cifar-10', 'cifar-100','celeb'], \
        "dataset parameter must be either 'mnist', 'svhn', 'cifar-10' or 'cifar-100'"

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_shape):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if dataset == 'mnist' or dataset == 'fashion_mnist':
        # ## LeNet-5
        x = Conv2D(6, (ks1, ks1), padding='same', kernel_initializer="he_normal", name='conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(12, (ks2, ks2), padding='same', kernel_initializer="he_normal", name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten()(x)

        x = Dense(100, kernel_initializer="he_normal", name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh', name='lid')(x)
        # x = Dropout(0.2)(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)

    elif dataset == 'svhn':
        # ## LeNet-5
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten()(x)

        x = Dense(512, kernel_initializer='he_normal', name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(128, kernel_initializer="he_normal", name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)
        # x = Dropout(0.2)(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)

    elif (dataset == 'cifar-10' or dataset == 'celeb') and bnn ==0 :
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)
        
        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        # Create model.
        model = Model(img_input, x)

    elif (dataset == 'cifar-10' or dataset == 'celeb') and bnn:
        
        dropout_rate = 0.2
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)

        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        # Create model.
        model = Model(img_input, x)
    
    elif dataset == "cifar-100":
        model = cifar100_resnet(depth=7, num_classes = num_classes)


    return model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

