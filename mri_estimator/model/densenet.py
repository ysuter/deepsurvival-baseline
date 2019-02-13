import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

# Hyperparameter
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2
SCALE = 1.0

# Label(s), 1 for regression
class_num = 1

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME', use_bias=True)
        return network

def Global_Average_Pooling(x):
    return tf.reduce_mean(x, name='Global_avg_pooling')


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=4)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



# class DenseNet():
#     def __init__(self, x, nb_blocks, nb_blocks, filters, is_train):
#         self.nb_blocks = nb_blocks
#         self.filters = filters
#         self.training = is_train
#         self.model = self.Dense_net(x)

def bottleneck_layer(filters, x, scope, is_train):
    # print(x)
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=is_train, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=4 * filters, kernel=[1,1,1], layer_name='layer_'+scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=is_train)

        x = Batch_Normalization(x, training=is_train, scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[3,3,3], layer_name='layer_'+scope+'_conv2')
        x = Drop_out(x, rate=dropout_rate, training=is_train)

        # print(x)

        return x

def transition_layer(filters, x, is_train, scope):
    with tf.name_scope(scope):

        x = Batch_Normalization(x, training=is_train, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[1,1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=is_train)
        x = Average_pooling(x, pool_size=[2,2,2], stride=2)

        return x

def dense_block(filters, input_x, nb_layers, is_train, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(filters, input_x, is_train=is_train, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(filters, x, is_train=is_train, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = Concatenation(layers_concat)

        return x

def Dense_net(features):

    with tf.variable_scope('NET'):
        input_layer = features["x"] / SCALE
        # output_layer = features["y"]
        is_train = features["is_train"]
        data = input_layer  # tf.reshape(input_layer, [-1, 256, 256, 256, 1])
        filters = 12
        nb_blocks = nb_block

        x = conv_layer(data, filter=2 * filters, kernel=[7, 7, 7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3, 3, 3], stride=2)



        for i in range(nb_blocks) :
            # 6 -> 12 -> 48
            x = dense_block(filters, input_x=x, nb_layers=4, is_train=is_train, layer_name='dense_'+str(i))
            x = transition_layer(filters, x, is_train, scope='trans_'+str(i))


        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """

        x = dense_block(filters, input_x=x, nb_layers=32, is_train=is_train, layer_name='dense_final')
        print(x.shape)
        print("########")
        # 100 Layer
        x = Batch_Normalization(x, training=is_train, scope='linear_batch')
        x_flat = tf.reshape(x, [-1, 9 * 14 * 14 * 396])
        dropout_flat = tf.layers.dropout(x_flat, rate=0.4, training=is_train)
        fc1 = tf.layers.dense(inputs=dropout_flat, units=100, use_bias=True, activation=tf.nn.relu, name='layer4_fc')
        dropout_fc1 = tf.layers.dropout(fc1, rate=0.2, training=is_train)
        fc2 = tf.layers.dense(inputs=dropout_fc1, units=20, use_bias=True, activation=None, name='layer5_fc')
        dropout_fc2 = tf.layers.dropout(fc2, rate=0, training=is_train)
        fc3 = tf.layers.dense(inputs=dropout_fc2, units=1, use_bias=True, activation=None, name='layer6_fc')
        ret = tf.identity(fc3, name='model')
        # x = Relu(x)
        # x = Global_Average_Pooling(x)
        print(x.shape)
        # x = flatten(x)
        #x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
    return ret