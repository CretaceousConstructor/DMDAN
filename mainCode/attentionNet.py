
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.layers import Conv2D

def ATCNet(n_classes, in_chans = 22, in_samples = 1125, n_windows = 3, attention = None,
           eegn_F1 = 16, eegn_D = 2, eegn_kernelSize = 64, eegn_poolSize = 8, eegn_dropout=0.3,
           tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3,
           tcn_activation = 'elu', fuse = 'average'):


def conv_block_original_to_seq(input, channel_size, F1, D, kernLength, poolSize,cls_num, is_training=True, reuse=False):
    #input形式参数值：input_layer = tf.placeholder(shape=[None, channel_size, time_size, depth_size], dtype=tf.float32) depth_size没有说明通常是1
    #函数调用方法：predict, prob, feat, net2 = generator(input_layer, channel_size, cls_num)
    #input:需要做卷积的输入，它要求是一个Tensor,具有[batch_size, in_height, in_width, in_channels]

    F2= F1*D
    Conv2D(F1,(kernLength,1),padding = 'same',data_format='channels_last',use_bias=False,activation=None,name="conv1")(input)


