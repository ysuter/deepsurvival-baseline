import tensorflow as tf
import numpy as np

SCALE = 1.0


def autoencoder(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        is_train = features["is_train"]
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=144,
            kernel_size=[11, 11],
            strides=[4, 4],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='enc_conv1')
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=72,
            kernel_size=[3, 3],
            strides=[1,1],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name = 'enc_conv2')
        pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2, name='enc_pool2')
        pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 192])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.2, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=8928, use_bias=True, activation=tf.nn.relu, name='layer4_fc')

        enc = tf.identity(fc1, name='encmodel')
        conv3  = tf.layers.conv2d(
            inputs=enc,
            filters=72,
            kernel_size=[3, 3],
            strides=[1,1],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name = 'enc_conv2')


def dencoder(encfeatures):
    with tf.variable_scope('NET'):
        input_layer = encfeatures
        is_train = encfeatures["is_train"]
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=144,
            kernel_size=[11, 11],
            strides=[4, 4],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='enc_conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, name='enc_pool1')
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=72,
            kernel_size=[3, 3],
            strides=[1,1],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            name = 'enc_conv2')
        pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2, name='enc_pool2')
        pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 192])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.2, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=8928, use_bias=True, activation=tf.nn.relu, name='layer4_fc')

        enc = tf.identity(fc1, name='model')

    return enc

