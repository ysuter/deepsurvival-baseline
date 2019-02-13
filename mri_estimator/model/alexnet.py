import tensorflow as tf
import numpy as np

SCALE = 1.0

def conv_net_alexnet_modV7_3d(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        is_train = features["is_train"]
        data = input_layer
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=64,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=36,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=36,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name = 'layer3_conv')
        print(conv3.shape)
        dropout_conv3 = tf.layers.dropout(conv3, rate=0.2, training=is_train)
        pool3 = tf.layers.max_pooling3d(inputs=dropout_conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 36])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=100, use_bias=True, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        fc2 = tf.layers.dense(inputs=dropout_fc1, units=20, use_bias=True, activation=None, kernel_initializer=tf.variance_scaling_initializer, name='layer5_fc')
        dropout_fc2 = tf.layers.dropout(fc2, rate=0.4, training=is_train)
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, kernel_initializer=tf.variance_scaling_initializer, name='layer6_fc')

        # conv3out = tf.identity(conv3, name='conv3out')
        fc100 = tf.identity(fc1, name='fc100')
        fc20 = tf.identity(fc2, name='fc20')
        # print(fc20.shape)

        ret = tf.identity(fc3, name='model')

        predictions = {
            "reg": ret,
            # "conv3out": conv3out,
            "fc100": fc100,
            "fc20": fc20
        }

    return predictions


def conv_net_alexnet_modV7_3d_demogr(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"]
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=144,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        dropout_conv1 = tf.layers.dropout(conv1, rate=0.1, training=is_train)
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=dropout_conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=192,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer2_conv')
        dropout_conv2 = tf.layers.dropout(conv2, rate=0.1, training=is_train)
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=dropout_conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=121,
            kernel_size=[3, 3, 3],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            name = 'layer3_conv')
        dropout_conv3 = tf.layers.dropout(conv3, rate=0.1, training=is_train)
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=dropout_conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 121])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=100, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        fc2 = tf.layers.dense(inputs=dropout_fc1, units=6*24, use_bias=True, activation=tf.nn.relu, name='layer5_fc')
        dropout_fc2 = tf.layers.dropout(fc2, rate=0.4, training=is_train)
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')
        ret = tf.identity(fc3, name='model')

    return ret

def conv_net_alexnet_modV7_3d_demogr2(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"] / 100
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=81,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=192,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=192,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 192])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=3*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        print(demographics.shape)
        print("%%%%%%%%%%%%%%%%%%%%%")
        fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc2 = tf.layers.dense(inputs=fc1_demographics, units=6*24, use_bias=True, activation=None, name='layer5_fc')
        #print(fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        #print(dropout_fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')
        #print(fc3.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        ret = tf.identity(fc3, name='model')

    return ret

def smallnet_3d_demogr(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"] # / 100
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=64,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 128])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=3*32, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        print(demographics.shape)
        print("%%%%%%%%%%%%%%%%%%%%%")
        fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc2 = tf.layers.dense(inputs=fc1_demographics, units=6*32, use_bias=True, activation=None, name='layer5_fc')
        #print(fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        #print(dropout_fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')
        #print(fc3.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        ret = tf.identity(fc3, name='model')

    return ret



def conv_net_alexnet_modV7_3d_nodemogr_smallkernel(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        #demographics = features["d"]
        is_train = features["is_train"]
        data = input_layer
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=64,
            kernel_size=[7, 7, 7],
            strides=[1, 1, 1],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=32,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=False,
            activation=tf.nn.relu,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 9 * 14 * 14 * 32])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=6*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        #fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #fc2 = tf.layers.dense(inputs=fc1_demographics, units=6*24, use_bias=True, activation=None, name='layer5_fc')
        fc2 = tf.layers.dense(inputs=fc1, units=6*24, use_bias=True, activation=None, name='layer5_fc')
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')

        ret = tf.identity(fc3, name='model')

    return ret


# def conv_net_alexnet_modV7_3d(features):
#     with tf.variable_scope('NET'):
#         input_layer = features["x"] / SCALE
#         output_layer = features["y"]
#         is_train = features["is_train"]
#         data = input_layer
#         conv1 = tf.layers.conv3d(
#             inputs=data,
#             filters=64,
#             kernel_size=[5, 5, 5],
#             strides=[2, 2, 2],
#             padding='same',
#             use_bias=True,
#             activation=tf.nn.relu,
#             name='layer1_conv')
#         print(conv1.shape)
#         dropout_conv1 = tf.layers.dropout(conv1, rate=0.2, training=is_train)
#         pool1 = tf.layers.max_pooling3d(inputs=dropout_conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
#         print(pool1.shape)
#         conv2 = tf.layers.conv3d(
#             inputs=pool1,
#             filters=64,
#             kernel_size=[5, 5, 5],
#             strides=[2, 2, 2],
#             padding='same',
#             use_bias=False,
#             activation=tf.nn.relu,
#             name='layer2_conv')
#         print(conv2.shape)
#         dropout_conv2 = tf.layers.dropout(conv2, rate=0.2, training=is_train)
#         pool2 = tf.layers.max_pooling3d(inputs=dropout_conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
#         print(pool2.shape)
#         conv3 = tf.layers.conv3d(
#             inputs=pool2,
#             filters=64,
#             kernel_size=[3, 3, 3],
#             padding='same',
#             use_bias=False,
#             activation=tf.nn.relu,
#             name = 'layer3_conv')
#         print(conv3.shape)
#         dropout_conv3 = tf.layers.dropout(conv3, rate=0.2, training=is_train)
#         pool3 = tf.layers.max_pooling3d(inputs=dropout_conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
#         print(pool3.shape)
#         pool3_flat = tf.reshape(pool3, [-1, 4 * 6 * 6 * 64])
#         dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
#         fc1 = tf.layers.dense(inputs=dropout_pool3, units=100, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
#         dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
#         fc2 = tf.layers.dense(inputs=dropout_fc1, units=20, use_bias=True, activation=None, name='layer5_fc')
#         dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
#         fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')
#
#         ret = tf.identity(fc3, name='model')
#
#     return ret


def conv_net_alexnet_CLASS_3d_demogr(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"] # / 100
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=81,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=64,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 64])
        print(pool3_flat.shape)
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=6*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        #print(demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc2 = tf.layers.dense(inputs=fc1_demographics, units=6*24, use_bias=True, activation=None, name='layer5_fc')
        #print(fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        #print(dropout_fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=3, use_bias=True, activation=None, name='layer6_fc')
        print(fc3.shape)
        print("%%%%%%%%%%%%%%%%%%%%%")
        logits = tf.identity(fc3, name='model')
        print(logits)
        # Logits Layer
        #logits = tf.layers.dense(inputs=fc3, units=3)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.one_hot(tf.argmax(input=logits, axis=1),3, on_value=1, off_value = 0),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            "logits": logits
        }

    return predictions


def conv_net_alexnet_CLASS_3d_demogr_TANH(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"] # / 100
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=81,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.tanh,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=True,
            activation=tf.nn.tanh,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=64,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.tanh,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 64])
        print(pool3_flat.shape)
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=6*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        #print(demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc2 = tf.layers.dense(inputs=fc1_demographics, units=6*24, use_bias=True, activation=None, name='layer5_fc')
        #print(fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        #print(dropout_fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=3, use_bias=True, activation=None, name='layer6_fc')
        print(fc3.shape)
        print("%%%%%%%%%%%%%%%%%%%%%")
        logits = tf.identity(fc3, name='model')
        print(logits)
        # Logits Layer
        #logits = tf.layers.dense(inputs=fc3, units=3)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.one_hot(tf.argmax(input=logits, axis=1),3, on_value=1, off_value = 0),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            "logits": logits
        }

    return predictions

def conv_net_alexnet_CLASS_3d_demogr_2CONV(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"] # / 100
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=81,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        pool2_flat = tf.reshape(pool2, [-1, 6 * 9 * 9 * 64])
        print(pool2_flat.shape)
        dropout_pool2 = tf.layers.dropout(pool2_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool2, units=6*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        #print(demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc2 = tf.layers.dense(inputs=fc1_demographics, units=6*24, use_bias=True, activation=None, name='layer5_fc')
        #print(fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        #print(dropout_fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=3, use_bias=True, activation=None, name='layer6_fc')
        print(fc3.shape)
        print("%%%%%%%%%%%%%%%%%%%%%")
        logits = tf.identity(fc3, name='model')
        print(logits)
        # Logits Layer
        #logits = tf.layers.dense(inputs=fc3, units=3)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.one_hot(tf.argmax(input=logits, axis=1),3, on_value=1, off_value = 0),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            "logits": logits
        }

    return predictions


def conv_net_alexnet_CLASS_3d_demogr_ReLUsmall(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        demographics = features["d"] # / 100
        print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=36,
            kernel_size=[11, 11, 11],
            strides=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.tanh,
            name='layer1_conv')
        print(conv1.shape)
        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 3, 3], strides=2, name='layer1_pool')
        print(pool1.shape)
        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=25,
            kernel_size=[5, 5, 5],
            strides=[2, 2, 2],
            padding='same',
            use_bias=True,
            activation=tf.nn.tanh,
            name='layer2_conv')
        print(conv2.shape)
        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2, name='layer2_pool')
        print(pool2.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool2,
            filters=16,
            kernel_size=[3, 3, 3],
            padding='same',
            use_bias=True,
            activation=tf.nn.tanh,
            name = 'layer3_conv')
        print(conv3.shape)
        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2, name='layer3_pool')
        print(pool3.shape)
        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 16])
        print(pool3_flat.shape)
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=4*64, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        #print(demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc1_demographics = tf.concat([dropout_fc1, demographics], axis=1)
        #print(fc1_demographics.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc2 = tf.layers.dense(inputs=fc1_demographics, units=4*24, use_bias=True, activation=None, name='layer5_fc')
        #print(fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        #print(dropout_fc2.shape)
        #print("%%%%%%%%%%%%%%%%%%%%%")
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=3, use_bias=True, activation=None, name='layer6_fc')
        print(fc3.shape)
        print("%%%%%%%%%%%%%%%%%%%%%")
        logits = tf.identity(fc3, name='model')
        print(logits)
        # Logits Layer
        #logits = tf.layers.dense(inputs=fc3, units=3)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.one_hot(tf.argmax(input=logits, axis=1),3, on_value=1, off_value = 0),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
            "logits": logits
        }

    return predictions

def regnet_conv_deepfeat(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        #demographics = features["d"]
        #print("demographics: " +str(demographics.shape))
        is_train = features["is_train"]
        data = input_layer  #tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        conv1 = tf.layers.conv3d(
            inputs=data,
            filters=64,
            kernel_size=[3, 3, 3],
            strides=[1, 1, 1],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer1_conv')
        dropout_conv1 = tf.layers.dropout(conv1, rate=0.1, training=is_train)
        print(conv1.shape)
        conv2 = tf.layers.conv3d(
            inputs=dropout_conv1,
            filters=64,
            kernel_size=[3, 3, 3],
            strides=[1, 1, 1],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer2_conv')
        dropout_conv2 = tf.layers.dropout(conv2, rate=0.1, training=is_train)
        print(conv2.shape)
        pool1 = tf.layers.max_pooling3d(inputs=dropout_conv2, pool_size=[5, 5, 5], strides=3, name='layer1_pool')
        print(pool1.shape)
        conv3 = tf.layers.conv3d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3, 3],
            strides=[1, 1, 1],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer3_conv')
        dropout_conv3 = tf.layers.dropout(conv3, rate=0.1, training=is_train)
        print(conv3.shape)
        conv4 = tf.layers.conv3d(
            inputs=dropout_conv3,
            filters=64,
            kernel_size=[3, 3, 3],
            strides=[1, 1, 1],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer4_conv')
        dropout_conv4 = tf.layers.dropout(conv4, rate=0.1, training=is_train)
        print(conv4.shape)
        pool2 = tf.layers.max_pooling3d(inputs=dropout_conv4, pool_size=[5, 5, 5], strides=3, name='layer1_pool')
        print(pool2.shape)
        conv5 = tf.layers.conv3d(
            inputs=pool2,
            filters=64,
            kernel_size=[3, 3, 3],
            strides=[1, 1, 1],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer5_conv')
        dropout_conv5 = tf.layers.dropout(conv5, rate=0.1, training=is_train)
        print(conv5.shape)
        conv6 = tf.layers.conv3d(
            inputs=dropout_conv5,
            filters=64,
            kernel_size=[3, 3, 3],
            strides=[1, 1, 1],
            padding='valid',
            use_bias=True,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer,
            name='layer6_conv')
        dropout_conv6 = tf.layers.dropout(conv5, rate=0.1, training=is_train)
        print(conv5.shape)
        pool3 = tf.layers.max_pooling3d(inputs=dropout_conv6, pool_size=[5, 5, 5], strides=3, name='layer1_pool')
        print(pool3.shape)


        pool3_flat = tf.reshape(pool3, [-1, 2 * 4 * 4 * 121])
        dropout_pool3 = tf.layers.dropout(pool3_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_pool3, units=100, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.4, training=is_train)
        print(dropout_fc1.shape)
        fc2 = tf.layers.dense(inputs=dropout_fc1, units=20, use_bias=True, activation=None, name='layer5_fc')
        dropout_fc2 = tf.layers.dropout(fc2, rate=0.4, training=is_train)
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer6_fc')

        fc100 = tf.identity(fc1, name='fc100')
        fc20 = tf.identity(fc2, name='fc20')

        ret = tf.identity(fc3, name='model')

        predictions = {
            "reg": ret,
            "fc100": fc100,
            "fc20": fc20
        }

    return predictions
