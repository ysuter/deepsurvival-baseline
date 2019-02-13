import tensorflow as tf

SCALE = 1.0


def block_cole(layer_num, input, num_kernels, is_train):
    conv1 = tf.layers.conv3d(
        inputs=input,
        filters=num_kernels,
        kernel_size=[3, 3, 3],
        strides=[1, 1, 1],
        padding='same',
        use_bias=True,
        activation=tf.nn.relu,
        name='layer{}_aconv'.format(layer_num))
    print(conv1.shape)
    conv2 = tf.layers.conv3d(
        inputs=conv1,
        filters=num_kernels,
        kernel_size=[3, 3, 3],
        strides=[1, 1, 1],
        padding='same',
        use_bias=True,
        activation=None,
        name='layer{}_bconv'.format(layer_num))
    print(conv2.shape)
    bn = tf.nn.relu(tf.layers.batch_normalization(conv2, training=is_train))
    pool = tf.layers.max_pooling3d(inputs=bn, pool_size=[2, 2, 2], strides=2, name='layer{}_pool'.format(layer_num))
    print(pool.shape)

    return pool

def conv_net_cole(features):
    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        output_layer = features["y"]
        is_train = features["is_train"]
        data = input_layer  # tf.reshape(input_layer, [-1, 256, 256, 256, 1])

        l1 = block_cole(1, data, 8, is_train)
        l2 = block_cole(2, l1, 16, is_train)
        l3 = block_cole(3, l2, 32, is_train)
        l4 = block_cole(4, l3, 64, is_train)
        l5 = block_cole(5, l4, 128, is_train)

        pool_flat = tf.reshape(l5, [-1, 4 * 7 * 7 * 128])
        dropout_pool = tf.layers.dropout(pool_flat, rate=0.4, training=is_train, name='dropout_pool')
        print("now fc...")
        fc1 = tf.layers.dense(inputs=dropout_pool, units=100, use_bias=True, activation=None, name='layer1_fc')
        fc2 = tf.layers.dense(inputs=fc1, units=20, use_bias=True, activation=None, name='layer2_fc')
        fc3 = tf.layers.dense(inputs=fc2, units=output_layer.shape[1], use_bias=True, activation=None, name='layer3_fc')

        ret = tf.identity(fc3, name='model')

    return ret