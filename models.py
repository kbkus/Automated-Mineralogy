
import numpy as np
import tensorflow as tf
from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict
import os
from config import imshape, n_classes, n_epochs

def model_orig(pretrained=False):

    img_input = tf.keras.Input(shape=(imshape))

    # define encoder layers
    # conv1 and conv2 contain intermediate the encoder outputs
    # which will be used by the decoder
    # pool2 is the final output of the encoder

    # two convolution layers and one pooling layer, which downsamples image by a factor of 2
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = tf.keras.layers.Dropout(0.2)(conv1)
    conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.2)(conv2)
    conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)

    # decoder layers
    conv4 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(conv4)

    # concat intermediate encoder outputs with intermediate decoder outputs, which is the skip connection
    up1 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D((2,2))(conv4), conv3], axis=-1)
    conv5 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(up1)
    conv5 = tf.keras.layers.Dropout(0.2)(conv5)
    conv5 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv5)

    up2 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D((2,2))(conv5), conv2], axis=-1)
    conv6 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(up2)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)
    conv6 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv6)

    up3 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D((2,2))(conv6), conv1], axis=-1)
    conv7 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(up3)
    conv7 = tf.keras.layers.Dropout(0.2)(conv7)
    conv7 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(conv7)

    # get output with proper number of classes
    out = tf.keras.layers.Conv2D(n_classes, (1,1), padding='same')(conv7)

    model = get_segmentation_model(img_input, out)
    model.train(
        train_images = '../dataset/xpl_train_images/',
        train_annotations = '../dataset/train_segmentation/',
        epochs = n_epochs)
    
    model.summary()
    
    return model