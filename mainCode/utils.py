#import tensorflow as tf
import tensorflow.compat.v1 as tf
#import torch
import numpy as np
from numpy import linalg
import random
from scipy.linalg import fractional_matrix_power

import matplotlib.pyplot as plt

#tf.disable_v2_behavior()

#def setup_seed(seed):
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.deterministic = True
#    np.random.seed(seed)
#    random.seed(seed)


def print_learning_rate(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            #if k is 'params':
            if k == 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)

def plot_feature_map(data, label, title):
    data_min, data_max = np.min(data, 0), np.max(data, 0)
    k = (15-(-15)) / (data_max-data_min)
    #data = (data - data_min) / (data_max - data_min)
    #temp_test = data[:, 0]


    data = -15.0 + k * (data-data_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        #plt.text(data[i, 0], data[i, 1], str(label[i]),
        #         color=plt.cm.Set1((label[i]+1.0) / 10.),
        #         fontdict={'weight': 'bold', 'size': 9})
        #label为 0 1 2 3分别对应左手（红色），右手（蓝色），双脚（绿色），舌头（紫色）
        plt.scatter(data[i, 0], data[i, 1],s=20, color=plt.cm.Set1((label[i]+1.0) / 10.),marker='o')
    #plt.xticks([])
    #plt.yticks([])
    plt.title(title)
    return fig







def PRelu(x, name='PRelu'):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha',shape=x.get_shape()[1:],dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=True)
        pos = tf.nn.relu(x)
        neg = -alpha * tf.nn.relu(-x)
        return pos + neg


def get_center_loss_raw(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


def get_center_loss(features, labels, alpha, num_classes, name):
   
    # 获取特征的维数，例如256维。这里features的维度是batch * 16
    len_features = features.get_shape()[1] #len_features = 16
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable(name, [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)#全部初始化为0
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])#维度只有1维，并且和batch大小一样，label中装的是独热编码转换成0 1 2 3的数值

    # 根据样本label,获取mini-batch中每一个样本对应的中心值。gather函数默认轴为axis = 0。labels可能的值（对于4分类）：0 1 2 3。center维度4 * 16，把labels的值作为center第一个维度的索引。
    #最后centers_batch的维度为[batch,16]，它来自centers（一个全0矩阵）
    centers_batch = tf.gather(centers, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差。features的维度也是[batch,16]，所以diff也是[batch,16].
    diff = centers_batch - features

    # 获取mini-batch中 出现的次数,了解原理请参考原文公式(4)
    # unique_label：如果是4分类，那就是0，1，2，3
    # unique_label：里面装的是下标，unique_label数组的下标
    # unique_count：对应数组unique_label中元素的数目
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    #用unique_idx作为index取unique_count中的元素。注意unique_idx和labels的长度一样，都是batch_size。
    #appear_times就是把labels中的每一个元素（可能为0，1，2，3）替换该元素在整个labels数组中出现的次数。
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])#比如从[1,2,3,4]变成[[1],[2],[3],[4]]
    #diff维度batch_size * 16 , appear_times维度 batch_size * 1，首先进行广播[[1],[2],[3],[4]] ----->  [[1,1,1,...],[2,2,2...],[3,3,3... ]...]
    diff = diff / tf.cast((1 + appear_times), tf.float32)#diff维度依然是batch_size * 16
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    # 计算loss
    with tf.control_dependencies([centers_update_op]):
#        loss = tf.nn.l2_loss(features - centers_batch)
        loss = tf.reduce_mean(tf.abs(features-centers_batch))
    return loss, centers








def euclidean_space_data_alignment(data):
    #EA处理之前的数据必须经过Band-pass filtering
    #一次trial的数据维度22*1000
    # X = 22 * 1000 , X^T = 1000 * 22, 协方差矩阵维度为 22 * 22

    # data维度:数据集A     400次实验 * 3采样点 * 1000时间点
    # data维度:数据集B     288次实验* 22样点 * 1000时间点
    data_result = data

    num_of_trials = data.shape[0]
    cov_matrix_dim = data.shape[1]
    r_bar = np.zeros((cov_matrix_dim,cov_matrix_dim))
    for i in range(num_of_trials):
        x = data[i]
        x_transpose = x.transpose()
        r_temp = np.dot(x,x_transpose)
        r_bar = r_bar + r_temp

    r_bar = r_bar / num_of_trials

    r_result = fractional_matrix_power(r_bar, -0.5)
    #  为特征值     为特征向量
    #eigen_values, eigen_vectors = linalg.eig(r_bar)
    #diagonal = np.diag(eigen_values**(-0.5))
    #r_result = eigen_vectors * diagonal * linalg.inv(eigen_vectors)

    for i in range(num_of_trials):
        x = data[i]
        data_result[i] = np.dot(r_result,x)

    return data_result



