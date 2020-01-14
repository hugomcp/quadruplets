import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def avgPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.avg_pool(x, ksize = [1, kHeight, kWidth, 1],
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


def vgg_like(x_inputs, keepPro, classNum):
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

#cur_shape = tf.shape(x_inputs).numpy()
# #conv_in = tf.reshape(pool1, [-1, cur_shape[0]/4 * cur_shape[1]/4 * 64])


def resnet_like(x_inputs, keepPro, classNum):
    conv1_1 = convLayer(x_inputs, 7, 7, 2, 2, 64, "conv1_1")
    pool1 = maxPoolLayer(conv1_1, 2, 2, 2, 2, "pool1")

    conv2_1 = convLayer(pool1, 3, 3, 1, 1, 64, "conv2_1")
    conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 64, "conv2_2")
    conc_1 = tf.concat([pool1, conv2_2], 1)

    conv3_1 = convLayer(conc_1, 3, 3, 1, 1, 64, "conv3_1")
    conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 64, "conv3_2")
    conc_2 = tf.concat([conc_1, conv3_2], 1)


    conv4_1 = convLayer(conc_2, 3, 3, 1, 1, 64, "conv4_1")
    conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 64, "conv4_2")
    conc_3 = tf.concat([conc_2, conv4_2], 1)

    conv5_1 = convLayer(conc_3, 3, 3, 2, 2, 128, "conv5_1")
    conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 128, "conv5_2")
    pool2 = avgPoolLayer(conc_3, 2, 2, 2, 2, "pool2")
    conc_4 = tf.concat([pool2, conv5_2], 1)

    conv6_1 = convLayer(conc_4, 3, 3, 1, 1, 128, "conv6_1")
    conv6_2 = convLayer(conv6_1, 3, 3, 1, 1, 128, "conv6_2")
    conc_5 = tf.concat([conc_4, conv6_2], 1)

    conv7_1 = convLayer(conc_5, 3, 3, 1, 1, 128, "conv7_1")
    conv7_2 = convLayer(conv7_1, 3, 3, 1, 1, 128, "conv7_2")
    conc_6 = tf.concat([conc_5, conv7_2], 1)

    conv8_1 = convLayer(conc_6, 3, 3, 1, 1, 128, "conv8_1")
    conv8_2 = convLayer(conv8_1, 3, 3, 1, 1, 128, "conv8_2")
    conc_7 = tf.concat([conc_6, conv8_2], 1)

    conv9_1 = convLayer(conc_7, 3, 3, 2, 2, 256, "conv9_1")
    conv9_2 = convLayer(conv9_1, 3, 3, 1, 1, 256, "conv9_2")
    pool3 = avgPoolLayer(conc_7, 2, 2, 2, 2, "pool3")
    conc_7 = tf.concat([pool3, conv9_2], 1)

    conv10_1 = convLayer(conc_7, 3, 3, 1, 1, 256, "conv10_1")
    conv10_2 = convLayer(conv10_1, 3, 3, 1, 1, 256, "conv10_2")
    conc_8 = tf.concat([conc_7, conv10_2], 1)

    conv11_1 = convLayer(conc_8, 3, 3, 1, 1, 256, "conv11_1")
    conv11_2 = convLayer(conv11_1, 3, 3, 1, 1, 256, "conv11_2")
    conc_9 = tf.concat([conc_8, conv11_2], 1)

    conv12_1 = convLayer(conc_9, 3, 3, 1, 1, 256, "conv12_1")
    conv12_2 = convLayer(conv12_1, 3, 3, 1, 1, 256, "conv12_2")
    conc_10 = tf.concat([conc_9, conv12_2], 1)

    conv13_1 = convLayer(conc_10, 3, 3, 1, 1, 256, "conv13_1")
    conv13_2 = convLayer(conv13_1, 3, 3, 1, 1, 256, "conv13_2")
    conc_11 = tf.concat([conc_10, conv13_2], 1)

    conv14_1 = convLayer(conc_11, 3, 3, 1, 1, 256, "conv14_1")
    conv14_2 = convLayer(conv14_1, 3, 3, 1, 1, 256, "conv14_2")
    conc_12 = tf.concat([conc_11, conv14_2], 1)

    conv15_1 = convLayer(conc_12, 3, 3, 2, 2, 512, "conv15_1")
    conv15_2 = convLayer(conv15_1, 3, 3, 1, 1, 512, "conv15_2")
    pool4 = avgPoolLayer(conc_12, 2, 2, 2, 2, "pool4")
    conc_13 = tf.concat([pool4, conv15_2], 1)

    conv16_1 = convLayer(conc_13, 3, 3, 1, 1, 256, "conv16_1")
    conv16_2 = convLayer(conv16_1, 3, 3, 1, 1, 256, "conv16_2")
    conc_14 = tf.concat([conc_13, conv16_2], 1)

    conv17_1 = convLayer(conc_14, 3, 3, 1, 1, 512, "conv17_1")
    conv17_2 = convLayer(conv17_1, 3, 3, 1, 1, 512, "conv17_2")
    conc_15 = tf.concat([conc_14, conv17_2], 1)

    pool5 = avgPoolLayer(conc_15, 2, 2, 2, 2, "pool5")

    fcIn = tf.reshape(pool5, [-1, 8 * 8 * 512])
    fc6 = fcLayer(fcIn, 8 * 8 * 512, 512, True, "fc6")
    drop1 = tf.nn.dropout(fc6, keepPro)

    fc8 = fcLayer(drop1, 256, classNum, False, "fc8")

    return(fc8)