
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float", initializer=tfc.layers.xavier_initializer())
        b = tf.get_variable("b", [outputD], dtype = "float", initializer=tfc.layers.xavier_initializer())
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding = "SAME"):
    """convolutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum], initializer=tfc.layers.xavier_initializer())
        b = tf.get_variable("b", shape = [featureNum], initializer=tfc.layers.xavier_initializer())
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        out = tf.nn.bias_add(featureMap, b)
    return tf.nn.relu(out, name=scope.name)


def vgg(x_inputs, keepPro, classNum):
    conv1_1 = convLayer(x_inputs, 3, 3, 1, 1, 64, "conv1_1" )
    conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
    conv1_3 = convLayer(conv1_2, 3, 3, 1, 1, 128, "conv1_3")
    pool1 = maxPoolLayer(conv1_3, 2, 2, 2, 2, "pool1")

    conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
    conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 256, "conv2_2")

    #poola = maxPoolLayer(conv2_2, 2, 2, 2, 2, "poola")

    conv2_3 = convLayer(conv2_2, 3, 3, 1, 1, 256, "conv2_3")
    conv2_4 = convLayer(conv2_3, 3, 3, 1, 1, 256, "conv2_4")
    pool2 = maxPoolLayer(conv2_4, 2, 2, 2, 2, "pool2")

    conv3_1 = convLayer(pool2, 3, 3, 1, 1, 512, "conv3_1")
    conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 512, "conv3_2")

    conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 512, "conv3_3")
    pool3 = maxPoolLayer(conv3_3, 2, 2, 2, 2, "pool3")


    #conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
    #conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
    #conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")

    #pool3 = tf.Print(pool3, ['POOL3: ',tf.shape(pool3)])


    fcIn = tf.reshape(pool3, [-1, 8*8*512])
    fc6 = fcLayer(fcIn, 8*8*512, 512, True, "fc6")
    drop1 = tf.nn.dropout(fc6, keepPro)

    fc7 = fcLayer(drop1, 512, 256, True, "fc7")
    drop2 = tf.nn.dropout(fc7, keepPro)

    fc8 = fcLayer(drop2, 256, classNum, False, "fc8")

    return fc8






