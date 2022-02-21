# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:14:08 2021

@author: simon
"""
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomContrast
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU, GlobalAveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation

def depth_block(model, strides):
    model.add(DepthwiseConv2D(3, strides=strides, activation='relu'))
    return model

def single_conv_block(model, filters):
    model.add(Conv2D(filters, 1, activation='relu'))
    return model

def combo_layer(model, filters, strides):
    model = depth_block(model,strides)
    model = single_conv_block(model, filters)
    return model

def MobileNet(input_shape=(224,224,3),n_classes = 1000):
    model = Sequential()
    model.add(Rescaling(1./255))
    model.add(RandomFlip(seed=1234))
    model.add(RandomRotation(seed=1234, factor=0.1))
    model.add(RandomContrast(0.1, seed=1234))
    model.add(Conv2D(32,3,strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(DepthwiseConv2D(3, strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(64, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(128, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(128, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(256, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(256, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(512, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    for _ in range(5):
      model.add(DepthwiseConv2D(3, strides=(1,1),padding='same'))
      model.add(BatchNormalization())
      model.add(ReLU(6))
      model.add(Conv2D(512, 1))
      model.add(BatchNormalization())
      model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(1024, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(DepthwiseConv2D(3, strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU(6))
    model.add(Conv2D(1024, 1))
    model.add(BatchNormalization())
    model.add(ReLU(6))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model