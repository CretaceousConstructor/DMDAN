
#import tensorflow as tf
import tensorflow.compat.v1 as tf



import scipy.io as sio
#import tensorflow.contrib.slim as slim
import tf_slim as slim
import numpy as np
from tensorflow import keras



def lrelu(x, leak=0.05, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def build_model(size_y, size_x, dim=1):
    # input layer
    img_input = keras.layers.Input(shape=(size_y, size_x, dim))

    # First convolution extracts 30 filters that are (size_y, 3)
    # Convolution is followed by max-pooling layer with a 1x10 window
    x = keras.layers.Conv2D(filters=30, kernel_size=(size_y, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0))(img_input)
    x = keras.layers.MaxPooling2D(1, 10)(x)

    # Convolution is followed by max-pooling layer with a 1x10 window
    x = keras.layers.Conv2D(filters=10, kernel_size=(1, 5), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.05))(x)
    x = keras.layers.MaxPooling2D(1, 3)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = keras.layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # x = keras.layers.MaxPooling1D(2)(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    # x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

    # Add a dropout rate of 0.5
    # x = keras.layers.Dropout(0.75)(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    # x = keras.layers.Dense(256, activation='sigmoid')(x)

    # Add a dropout rate of 0.5
    # x = keras.layers.Dropout(0.75)(x)
    # Create output layer with a single node and sigmoid activation
    output = keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0))(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = keras.models.Model(img_input, output)
    model.summary()
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=keras.optimizers.RMSprop(lr=0.001),
    #               metrics=['acc'])
    op1=keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=op1,
                  metrics=['acc'])
    return model














def spectrogram_net(input, channel_size, cls_num=1):
    # weights_regularizer = slim.l1_regularizer(0.01)
    weights_regularizer = slim.l1_regularizer(0.00001) # best 0.00001
    # weights_regularizer = None
    bn = slim.batch_norm

    net = slim.conv2d(input, 16, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer, normalizer_fn=bn)
    net = slim.max_pool2d(net,[1,7], stride=[1,7])


    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, 32, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred

def spectrogram_net2(input, channel_size, cls_num=1):
    # weights_regularizer = slim.l1_regularizer(0.01)
    weights_regularizer = slim.l1_regularizer(0.00001) # best 0.00001
    # weights_regularizer = None

    net = slim.conv2d(input, 16, [1, 10], weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 10],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 32, [1, 5],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 5],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])

    net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[1,2], stride=[1,2])


    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 128, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, 64, weights_regularizer=None)
    # net = slim.dropout(net)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred

def resnet(input, channel_size, cls_num=1):
    weights_regularizer = slim.l1_regularizer(0.00001)  # best 0.00001

    net = slim.conv2d(input, 64, [7,7],stride=2, weights_regularizer=weights_regularizer)
    net = slim.max_pool2d(net,[3,3],stride=2)

    x = net
    net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 64, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = net + x

    x = slim.conv2d(net, 128, [1,1], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 128, [3, 3], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 128, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = net + x

    x = slim.conv2d(net, 256, [1,1], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 256, [3, 3], stride=2, normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = slim.conv2d(net, 256, [3, 3], normalizer_fn=slim.batch_norm, weights_regularizer=weights_regularizer)
    net = net + x

    net = slim.avg_pool2d(net, [2,2])

    net = slim.flatten(net)

    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)


    return net, pred


def spectrogram_net_tst(input, channel_size, cls_num=1):
    # weights_regularizer = slim.l1_regularizer(0.01)
    weights_regularizer = slim.l2_regularizer(0.00001) # best

    net = slim.conv2d(input, 16, [3, 3],weights_regularizer = weights_regularizer)
    net = slim.conv2d(net, 16, [3, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[2,2], stride=[2,2])

    net = slim.conv2d(net, 32, [3, 3],weights_regularizer = weights_regularizer)
    net = slim.conv2d(net, 32, [3, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 3],weights_regularizer = weights_regularizer)
    net = slim.max_pool2d(net,[2,2], stride=[2,2])

    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.conv2d(net, 64, [1, 3],weights_regularizer = weights_regularizer)
    # net = slim.max_pool2d(net,[1,7], stride=[1,7])


    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=weights_regularizer)
    net = slim.fully_connected(net, 32, weights_regularizer=weights_regularizer)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred


def spectrogram_net_1d(input, channel_size, cls_num):

    weights_regularizer = None#slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [channel_size, 3], padding='VALID', weights_regularizer=weights_regularizer)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 10, [1, 5], padding='VALID',weights_regularizer=weights_regularizer)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.max_pool2d(net, [1, 2], stride=[1,2])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=weights_regularizer)
    # net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred


def tju_csp(input,channel_size, cls_num=1):
    net = slim.conv2d(input, 32, [1, 4], padding='VALID', normalizer_fn=slim.batch_norm)

    net = slim.conv2d(net, 7, [7, 1], padding='VALID', normalizer_fn=slim.batch_norm)

    net = slim.avg_pool2d(net, [4,1], stride=[4,1],padding='SAME')

    net = slim.flatten(net)

    net = slim.fully_connected(net, 2, activation_fn=None)

    pred = tf.nn.softmax(net)

    return net, pred


def signal(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    #net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred



def signal_siamese(input, channel_size, cls_num, is_training, reuse=False):



    weights_regularizer = slim.l2_regularizer(0.01)
    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='conv_time')
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])
    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='conv_spatial')
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])
    fc = slim.flatten(net)
    net1 = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='fc')
    # net3 = slim.fully_connected(net1, 32, reuse=reuse, scope='fc1', activation_fn=tf.nn.relu)
    # net4 = slim.fully_connected(net3, 32, reuse=reuse, scope='fc2', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net1, cls_num, activation_fn=None, reuse=reuse, scope='output')
    pred = tf.nn.softmax(net)
    # net4 = slim.fully_connected(net1, 8, reuse=reuse, scope='fea')

    return net, pred, net1


def signal_siamese_da(input, channel_size, cls_num, is_training=True, reuse=False):


    weights_regularizer = slim.l2_regularizer(0.01)
    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='g_conv_time')
    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='g_conv_spatial')

    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    #把网络输出打成1维，然后喂给全连接层
    fc = slim.flatten(net)

    net1 = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='g_fc')
    net2 = slim.fully_connected(net1, 32, reuse=reuse, scope='g_fc1')
    # net4 = slim.fully_connected(net3, 32, reuse=reuse, scope='fc2', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net2, cls_num, activation_fn=None, reuse=reuse, scope='g_output')
    pred = tf.nn.softmax(net)
    # net4 = slim.fully_connected(net1, 8, reuse=reuse, scope='fea')

    return net, pred, net1



def signal_siamese_da_fc64(input, channel_size, cls_num, is_training=True, reuse=False):

    #input形式参数值：input_layer = tf.placeholder(shape=[None, channel_size, time_size, depth_size], dtype=tf.float32)
    #函数调用方法：predict, prob, feat, net2 = generator(input_layer, channel_size, cls_num)
    #input:需要做卷积的输入图像，它要求是一个Tensor,具有[batch_size, in_height, in_width, in_channels]

    #L2正则化防止过拟合，0.01超参数
    weights_regularizer = slim.l2_regularizer(0.01)

    #=============================以下是特征提取器================================================
    #卷积神经网络，这里30是卷积核个数，[1,25]卷积核维度，padding方式选择VALID。
    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='g_conv_time')#提取时域信息

    #特征图normalization
    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='g_conv_spatial')#提取空域信息，卷积核大小和电极数目相同
    #均值池化
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    fc = slim.flatten(net)
    net1 = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='g_fc')
    #=============================以下是分类器================================================
    net2 = slim.fully_connected(net1, 16, reuse=reuse, scope='g_fc1',activation_fn=lrelu)
    # net4 = slim.fully_connected(net3, 32, reuse=reuse, scope='fc2', activation_fn=tf.nn.leaky_relu)
    net = slim.fully_connected(net1, cls_num, activation_fn=None, reuse=reuse, scope='g_output')
    pred = tf.nn.softmax(net)
    # net4 = slim.fully_connected(net1, 8, reuse=reuse, scope='fea')
    #第一个返回参数的输出维度是：4（4类）
    #第二个返回参数是4个类别的概率
    #第三个返回参数的输出维度是：64
    #第四个返回参数的输出维度是：16
    return net, pred, net1, net2







def signal_multitask(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    fc = slim.fully_connected(fc, 128)
    fc = slim.fully_connected(fc, 64)


    net1 = slim.fully_connected(fc, 64)
    net1 = slim.fully_connected(net1, 32)
    net1 = slim.fully_connected(net1, 9, activation_fn=None)


    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)


    pred = tf.nn.softmax(net)
    person = tf.nn.softmax(net1)



    return net, pred, net1, person


def signal_multitask_siamese(input, channel_size, cls_num, is_training, reuse=False):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None, reuse=reuse, scope='conv_time')
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm, reuse=reuse, scope='conv_spatial')
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    fc = slim.fully_connected(fc, 128, reuse=reuse, scope='fc1')
    fc = slim.fully_connected(fc, 64, reuse=reuse, scope='fc2')


    net1 = slim.fully_connected(fc, 64, reuse=reuse, scope='fc3')
    net1 = slim.fully_connected(net1, 32, reuse=reuse, scope='fc4')
    net1 = slim.fully_connected(net1, 9, activation_fn=None, reuse=reuse, scope='fc5')


    net = slim.fully_connected(fc, 64, weights_regularizer=None, reuse=reuse, scope='fc6')
    net_2 = slim.fully_connected(net, 32, reuse=reuse, scope='fc7')
    net = slim.fully_connected(net_2, cls_num, activation_fn=None, reuse=reuse, scope='fc8')


    pred = tf.nn.softmax(net)
    person = tf.nn.softmax(net1)



    return net, pred, net1, person, net_2


def signal_multitask_rerun(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)


    net1 = slim.fully_connected(fc, 32)
    net1 = slim.fully_connected(net1, 9, activation_fn=None)
    person = tf.nn.softmax(net1)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)



    return net, pred, net1, person






def signal_multitask_fusion(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net = slim.conv2d(input, 30, [1, 25], padding='VALID', weights_regularizer=None, activation_fn=None)
    # net = slim.conv2d(net, 16, [1, 3], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID',weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 32, [1, 3], padding='SAME')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])

    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)


    net1_1 = slim.fully_connected(fc, 32)
    net1 = slim.fully_connected(net1_1, 9, activation_fn=None)



    net = slim.fully_connected(fc, 64, weights_regularizer=None)

    net2 = tf.concat([net, net1_1], axis=1)

    net = slim.fully_connected(net2, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)

    pred = tf.nn.softmax(net)
    person = tf.nn.softmax(net1)

    return net, pred, net1, person

def signal_more(input, channel_size, cls_num, is_training):

    weights_regularizer = slim.l2_regularizer(0.01)

    net_1 = slim.conv2d(input, 10, [1, 25], stride=[1, 1], padding='SAME', weights_regularizer=None)
    net_2 = slim.conv2d(input, 10, [1, 15], stride=[1, 1], padding='SAME')
    net_3 = slim.conv2d(input, 10, [1, 35], stride=[1, 1], padding='SAME')
    # net = slim.max_pool2d(net, [1, 10],stride=[1,10])
    net = tf.concat([net_1, net_2, net_3], axis=3)

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID', weights_regularizer=None, normalizer_fn=slim.batch_norm)
    # net = slim.conv2d(net, 30, [1, 3], padding='VALID')
    net = slim.avg_pool2d(net, [1, 75], stride=[1,15])
    # net = slim.conv2d(net, 30, [1, 5], padding='VALID')
    # net = slim.conv2d(net, 64, [1, 3], padding='VALID')
    # net = slim.max_pool2d(net, [1, 5])

    fc = slim.flatten(net)

    net = slim.fully_connected(fc, 64, weights_regularizer=None)
    net = slim.fully_connected(net, 32)
    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred


def signal_dense(input, channel_size, cls_num, is_training):

    net = slim.conv2d(input, 30, [1, 25], padding='VALID')

    net = slim.conv2d(net, 30, [channel_size, 1], padding='VALID', normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training})

    net = slim.max_pool2d(net, [1, 3], stride=[1, 3])

    x = net

    net = slim.conv2d(net, 40, [1, 1], activation_fn=tf.nn.elu)

    net = slim.dropout(net, is_training=is_training)

    net = tf.concat([x, net], axis=-1)

    x = net

    net = slim.conv2d(net, 40,  [1, 1], activation_fn=tf.nn.elu)

    net = slim.dropout(net, is_training=is_training)

    net = tf.concat([x,net], axis=-1)

    net = slim.max_pool2d(net, [1, 3], [1, 3])

    net = slim.flatten(net)

    net = slim.fully_connected(net, 64)

    net = slim.fully_connected(net, cls_num, activation_fn=None)
    pred = tf.nn.softmax(net)

    return net, pred





def discriminator(fea, reuse=False):

    fc1 = slim.fully_connected(fea, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1')
    fc2 = slim.fully_connected(fc1, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2')
    fc3 = slim.fully_connected(fc2, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3')
    d_out_logits = slim.fully_connected(fc3, 1, activation_fn=None, reuse=reuse, scope='d_out')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out = tf.nn.sigmoid(d_out_logits)

    return d_out, d_out_logits


def dynamic_discriminator_four_cls(fea,classifier_out,reuse=False):
    #global_domain_discriminator
    fc1 = slim.fully_connected(fea, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1')
    fc2 = slim.fully_connected(fc1, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2')
    fc3 = slim.fully_connected(fc2, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3')
    d_out_logits_global = slim.fully_connected(fc3, 1, activation_fn=None, reuse=reuse, scope='d_out')
    #应用sigmoid函数可以将输出压缩至0～1的范围，成为一个二分类器
    d_out_global = tf.nn.sigmoid(d_out_logits_global)

    #local_domain_discriminator
    #===========用classifier_out鉴别器输出计算输入进每一个子鉴别器的概率，计算结果分别记作 prob_sub_classifier_0 - 4==============================
    #局部鉴别器总共有4个（4分类的缘故）,每一个负责 一个分类所有样本中 源领域和目标领域的mathcing
    #生成器的输出（实际上就是分类器的输出）可以表示一个样本进入哪个子鉴别器权重高
    #classifier_out 维度 ： [None , 4] 第一个维度是batch
    #fea 维度            ： [None , 64]


    prob_sub_classifier_0 = classifier_out[:,0]
    prob_sub_classifier_1 = classifier_out[:,1]
    prob_sub_classifier_2 = classifier_out[:,2]
    prob_sub_classifier_3 = classifier_out[:,3]


    #类别0样本中，源域和目标域做matching
    fc1_cls0 = slim.fully_connected(fea * prob_sub_classifier_0, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1_cls0')
    fc2_cls0 = slim.fully_connected(fc1_cls0, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2_cls0')
    fc3_cls0 = slim.fully_connected(fc2_cls0, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3_cls0')
    d_out_logits_cls0 = slim.fully_connected(fc3_cls0, 1, activation_fn=None, reuse=reuse, scope='d_out_cls0')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out_cls0 = tf.nn.sigmoid(d_out_logits_cls0)

    #类别1样本中，源域和目标域做matching
    fc1_cls1 = slim.fully_connected(fea * prob_sub_classifier_1, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1_cls1')
    fc2_cls1 = slim.fully_connected(fc1_cls1, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2_cls1')
    fc3_cls1 = slim.fully_connected(fc2_cls1, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3_cls1')
    d_out_logits_cls1 = slim.fully_connected(fc3_cls1, 1, activation_fn=None, reuse=reuse, scope='d_out_cls1')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out_cls1 = tf.nn.sigmoid(d_out_logits_cls1)


    #类别2样本中，源域和目标域做matching
    fc1_cls2 = slim.fully_connected(fea *prob_sub_classifier_2, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1_cls2')
    fc2_cls2 = slim.fully_connected(fc1_cls2, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2_cls2')
    fc3_cls2 = slim.fully_connected(fc2_cls2, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3_cls2')
    d_out_logits_cls2 = slim.fully_connected(fc3_cls2, 1, activation_fn=None, reuse=reuse, scope='d_out_cls2')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out_cls2 = tf.nn.sigmoid(d_out_logits_cls2)

    #类别3样本中，源域和目标域做matching
    fc1_cls3 = slim.fully_connected(fea * prob_sub_classifier_3, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1_cls3')
    fc2_cls3 = slim.fully_connected(fc1_cls3, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2_cls3')
    fc3_cls3 = slim.fully_connected(fc2_cls3, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3_cls3')
    d_out_logits_cls3 = slim.fully_connected(fc3_cls3, 1, activation_fn=None, reuse=reuse, scope='d_out_cls3')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out_cls3 = tf.nn.sigmoid(d_out_logits_cls3)

    #输出需要全局鉴别器的输出，同样也要输出子鉴别器的输出，从而能够同时计算两种鉴别器的损失


    return d_out_global, d_out_logits_global,d_out_logits_cls0,d_out_cls0,d_out_logits_cls1,d_out_cls1,d_out_logits_cls2,d_out_cls2,d_out_logits_cls3,d_out_cls3


def dynamic_discriminator_two_cls(fea,classifier_out,reuse=False):
    #global_domain_discriminator
    fc1 = slim.fully_connected(fea, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1')
    fc2 = slim.fully_connected(fc1, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2')
    fc3 = slim.fully_connected(fc2, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3')
    d_out_logits_global = slim.fully_connected(fc3, 1, activation_fn=None, reuse=reuse, scope='d_out')
    #应用sigmoid函数可以将输出压缩至0～1的范围，成为一个二分类器
    d_out_global = tf.nn.sigmoid(d_out_logits_global)

    #local_domain_discriminator
    #===========用classifier_out鉴别器输出计算输入进每一个子鉴别器的概率，计算结果分别记作 prob_sub_classifier_0 - 4==============================
    #局部鉴别器总共有4个（4分类的缘故）,每一个负责 一个分类所有样本中 源领域和目标领域的mathcing
    #生成器的输出（实际上就是分类器的输出）可以表示一个样本进入哪个子鉴别器权重高
    #classifier_out 维度 ： [None , 4] 第一个维度是batch
    #fea 维度            ： [None , 64]


    prob_sub_classifier_0 = classifier_out[:,0]
    prob_sub_classifier_1 = classifier_out[:,1]


    #类别0样本中，源域和目标域做matching
    fc1_cls0 = slim.fully_connected(fea * prob_sub_classifier_0, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1_cls0')
    fc2_cls0 = slim.fully_connected(fc1_cls0, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2_cls0')
    fc3_cls0 = slim.fully_connected(fc2_cls0, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3_cls0')
    d_out_logits_cls0 = slim.fully_connected(fc3_cls0, 1, activation_fn=None, reuse=reuse, scope='d_out_cls0')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out_cls0 = tf.nn.sigmoid(d_out_logits_cls0)

    #类别1样本中，源域和目标域做matching
    fc1_cls1 = slim.fully_connected(fea * prob_sub_classifier_1, 64, activation_fn=lrelu, reuse=reuse, scope='d_fc1_cls1')
    fc2_cls1 = slim.fully_connected(fc1_cls1, 32, activation_fn=lrelu, reuse=reuse, scope='d_fc2_cls1')
    fc3_cls1 = slim.fully_connected(fc2_cls1, 16, activation_fn=lrelu, reuse=reuse, scope='d_fc3_cls1')
    d_out_logits_cls1 = slim.fully_connected(fc3_cls1, 1, activation_fn=None, reuse=reuse, scope='d_out_cls1')
    #应用sigmoid函数可以将输出压缩至0～1的范围
    d_out_cls1 = tf.nn.sigmoid(d_out_logits_cls1)

    #输出需要全局鉴别器的输出，同样也要输出子鉴别器的输出，从而能够同时计算两种鉴别器的损失


    return d_out_global, d_out_logits_global,d_out_logits_cls0,d_out_cls0,d_out_logits_cls1,d_out_cls1


