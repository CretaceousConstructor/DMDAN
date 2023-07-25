import numpy as np
import scipy.io as sio
import net
import os
import time
import random
import sklearn
import shutil
import argparse
import ast
import utils
import warnings

from tensorflow import keras
from tensorflow.python.training import training_util
import tensorflow.compat.v1 as tf
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#import tensorflow.contrib.slim as slim
import tf_slim as slim
#import tensorboard as tb
import dataFile
import seaborn as sns
import pandas as pd
import loadData

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


physical_gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*10)]
)
logical_gpus = tf.config.list_logical_devices("GPU")


#禁用tensorflow v2的功能
tf.disable_v2_behavior()


#------------------------
config = tf.ConfigProto(allow_soft_placement=False)
#config.gpu_options.allow_growth = False
#config.gpu_options.per_process_gpu_memory_fraction = 0.5

#------------------------

#77777777777777777[]
##NumPy为了方便数据展示，采用科学计数法，但不太便于观察，所以取消它。
np.set_printoptions(suppress=True)
##对于未来特性更改的警告，直接忽略
warnings.simplefilter(action='ignore', category=FutureWarning)

##参数设置
## =================================== arg-parsing ========================================
parser = argparse.ArgumentParser()

##是否使用测试集，在之后的分支中，会用try block试着使用这个参数，如果这个参数没有定义，就不使用测试集
#parser.add_argument('--model', default='test')


parser.add_argument('--dataset', default='A')#选择B数据集
parser.add_argument('--subject', default=9, type=int)#选择第五个受害者作为目标域
#parser.add_argument('--lr', default=0.0002, type=float)#学习率
parser.add_argument('--lr', default=0.00009, type=float)#学习率
parser.add_argument('--max_epoch', default=200, type=int)#200个epoch
parser.add_argument('--batch_size', default=64, type=int)#batch大小
parser.add_argument('--exp_name', default='test')#实验名
parser.add_argument('--aug', default=False,type=ast.literal_eval)
parser.add_argument('--classifier', default='signal_da_fc64')#分类器选择signal_da_fc64
#parser.add_argument('--criteria', default='val_loss')#early stopping采用什么标准
parser.add_argument('--criteria', default='val_acc')
parser.add_argument('--loadN', default='1')
# parser.add_argument('--dataload', default='dataB')
parser.add_argument('--stop_tolerance', default=100, type=int)#early stopping
parser.add_argument('--data_len', default='data_0-4')
parser.add_argument('--w_adv', default='0.01',type=float)#超参数
parser.add_argument('--w_t', default='1', type=float)#超参数
parser.add_argument('--w_s', default='1', type=float)#超参数
parser.add_argument('--w_c', default='0.01', type=float)#超参数

args = parser.parse_args()

print(args)

## =================================== path setting ========================================

##不进行模型加载
load_model = None#'test_init'
##进行模型保存
save_model = True


# import pdb
# pdb.set_trace()

exp_name = args.exp_name

##如果没有在之前的参数中定义model，那么这里就会进入except分支
try:
    model = args.model
except:
#    #结果为：model = "s5"    s表示subject,受害者，5表示第五个
    model = 's' + str(args.subject)



##存放结果的文件夹
result_dir = 'Model_and_Result' + '/' + exp_name + '/' + model
##存放模型的文件夹
model_directory = result_dir + '/models'






## data path

##dataset：A或者B，首先选择B
dataset = args.dataset
## datapath： 第几个受害者，这里是5号
dataPath = args.subject

if tf.gfile.Exists(result_dir):
    tf.gfile.DeleteRecursively(result_dir)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

##把下面3个py文件和当前文件拷贝到结果文件夹中
shutil.copy(__file__, result_dir)
shutil.copy('net.py', result_dir)
shutil.copy('dataFile.py', result_dir)
shutil.copy('loadData.py', result_dir)

# os.system('cp {} {}'.format(__file__, result_dir))
# os.system('cp {} {}'.format('net.py', result_dir))

##写入代替创建，相当于创建了一个log文件
with open(result_dir + '/training_log.txt', 'w') as f:
    f.close()


## =========================== parameters setting =========================
lr = args.lr
# beta1 = 0.5
max_epoch = args.max_epoch
batch_size = args.batch_size
stop_criteria = args.criteria


# ================================   data loading and preprocessing  ==========================================================

if dataset == 'A':
    if dataPath < 6:
        path = './data/dataA/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
    else:
        path = './data/dataA/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'
elif dataset == 'B':
    if dataPath < 8:
        path = './data/dataB/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
    else:
        path = './data/dataB/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'

# trnData, trnLabel, tstData, tstLabel = dataFile.dataB_valSet(path, loadN=args.loadN, aug=args.aug)

##数据集读入
# sourceData, sourceLabel, targetData, targetLabel, tstData, tstLabel = dataFile.data_multitask_da(aug=args.aug, dataset=dataset, loadN=args.loadN, subject=args.subject)
sourceData, sourceLabel, targetData, targetLabel, tstData, tstLabel = dataFile.bciiv2a_multitask_da(subject=args.subject, dataset=args.dataset, data_len=args.data_len, ea_preprocess=True)

##sourcedata的维度：4608笔数据，维度22*625
##tagetdata的维度：288笔数据，维度22*625
##testdata测试数据，288笔，维度22*625，也是来自目标域的受害者数据
# trnData = trnData.astype('float32')
# tstData = tstData.astype('float32')

##打印标签种类，并且统计各个标签对应的数目
print('Source set label and proportion:\t', np.unique(sourceLabel, return_counts=True))
print('Target set label and proportion:\t', np.unique(targetLabel, return_counts=True))
print('Val   set label and proportion:\t', np.unique(tstLabel, return_counts=True))

##类别数目4类
cls_num = len(np.unique(targetLabel))


##数据的维度是：A数据集：(288, 22, 1000)22代表22个头部采样点，625是实验时间点，288样本数
##            B数据集：(400,3, 1000)
dataSize = targetData.shape
channel_size = dataSize[1]
time_size = dataSize[2]
try:
    depth_size = dataSize[3]
except:
    depth_size = 1

targetLabel = keras.utils.to_categorical(targetLabel, num_classes=cls_num)
sourceLabel = keras.utils.to_categorical(sourceLabel, num_classes=cls_num)
tstLabel = keras.utils.to_categorical(tstLabel, num_classes=cls_num)

## ============================= model function===============================

##选择生成器和辨别器函数
if args.classifier == 'signal':
    generator = net.signal_da#spectrogram_net
elif args.classifier == 'signal_more':
    generator = net.signal_more
elif args.classifier == 'signal_da':
    generator = net.signal_siamese_da
elif args.classifier == 'signal_da_fc64':
    generator = net.signal_siamese_da_fc64
    #generator = net.FSNAL_net_fc64



#discriminator = net.discriminator
if cls_num == 2:
    discriminator = net.dynamic_discriminator_two_cls
elif cls_num == 4:
    discriminator = net.dynamic_discriminator_four_cls
else:
    raise Exception("Unsupported number of classes.")
# ===================================== model definition ====================================

tf.reset_default_graph()

#计算daf
#ω is initialized as 1 in the first epoch
DAF_flat_value = 1.0
dynamic_adversarial_factor = tf.Variable(DAF_flat_value,shape=(),trainable=False)
mid_vari_DAF  = tf.placeholder(shape=(), dtype=tf.float32)
mid_assign_action =tf.assign(dynamic_adversarial_factor, mid_vari_DAF)

##None表示任意维，是批次中样本的个数
input_layer = tf.placeholder(shape=[None, channel_size, time_size, depth_size], dtype=tf.float32)
label_layer = tf.placeholder(shape=[None, cls_num], dtype=tf.float32)
# is_training = tf.placeholder(shape=[], dtype=tf.bool)
input_layer_s = tf.placeholder(shape=[None, channel_size, time_size, depth_size], dtype=tf.float32)
label_layer_s = tf.placeholder(shape=[None, cls_num], dtype=tf.float32)



##把输入扔进生成器中
##第一个返回参数的输出维度是：4（4类）没有经过softmax的值
##第二个返回参数是4个类别的概率
##第三个返回参数的输出维度是：64
##第四个返回参数的输出维度是：16

#predict：softmax之前的输出，prob：经过softmax的概率输出，feat表示feature为特征提取器的输出。
predict, prob, feat, net2 = generator(input_layer, channel_size, cls_num)
predict_s, prob_s, feat_s, net2_s = generator(input_layer_s, channel_size, cls_num, reuse=True)


#d_global_out, d_global_logits,_,_,_,_,_,_,_,_ = discriminator(feat,prob)


if cls_num == 4:
    d_out_global, d_out_logits_global,d_out_logits_cls0,d_out_cls0,d_out_logits_cls1,d_out_cls1,d_out_logits_cls2,d_out_cls2,d_out_logits_cls3,d_out_cls3 = discriminator(feat,prob)
    d_out_global_s, d_out_logits_global_s,d_out_logits_cls0_s,d_out_cls0_s,d_out_logits_cls1_s,d_out_cls1_s,d_out_logits_cls2_s,d_out_cls2_s,d_out_logits_cls3_s,d_out_cls3_s = discriminator(feat_s,prob_s,reuse=True)
elif cls_num == 2:
    d_out_global, d_out_logits_global,d_out_logits_cls0,d_out_cls0,d_out_logits_cls1,d_out_cls1 = discriminator(feat,prob)
    d_out_global_s, d_out_logits_global_s,d_out_logits_cls0_s,d_out_cls0_s,d_out_logits_cls1_s,d_out_cls1_s = discriminator(feat_s,prob_s,reuse=True)
else:
    raise Exception("Unsupported number of classes.")

#Dx, Dx_logits = discriminator(feat)#目标领域
#Dg, Dg_logits = discriminator(feat_s, reuse=True) #fake 源领域送进鉴别器，鉴别特征是来自源领域还是目标领域。

##下面这句没作用，因为返回的值没有被使用
#tf.argmax(label_layer, axis=1)

## ============================ loss function and optimizer =======================

#鉴别器误差。鉴别器尽可能使得源领域和目标领域输入 feature 提取器后的输出尽可能地接近。
    #    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx_logits, labels=tf.ones_like(Dx)))
    #    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logits, labels=tf.zeros_like(Dg)))
    #    d_loss = 0.5*d_loss_real + 0.5*d_loss_fake
    #the discriminator is trained on a binary domain label set Z = {0, 1}, in which the domain label is 1 for target data and 0 for the source samples.
    #d_loss = tf.reduce_mean(tf.square(Dx_logits - 1) + tf.square(Dg_logits)) / 2




d_loss_global = tf.reduce_mean(tf.square(d_out_logits_global - 1) + tf.square(d_out_logits_global_s)) / 2

if cls_num == 4:
    d_loss_cls0 = tf.reduce_mean(tf.square(d_out_logits_cls0 - 1) + tf.square(d_out_logits_cls0_s)) / 2
    d_loss_cls1 = tf.reduce_mean(tf.square(d_out_logits_cls1 - 1) + tf.square(d_out_logits_cls1_s)) / 2
    d_loss_cls2 = tf.reduce_mean(tf.square(d_out_logits_cls2 - 1) + tf.square(d_out_logits_cls2_s)) / 2
    d_loss_cls3 = tf.reduce_mean(tf.square(d_out_logits_cls3 - 1) + tf.square(d_out_logits_cls3_s)) / 2
elif cls_num == 2:
    d_loss_cls0 = tf.reduce_mean(tf.square(d_out_logits_cls0 - 1) + tf.square(d_out_logits_cls0_s)) / 2
    d_loss_cls1 = tf.reduce_mean(tf.square(d_out_logits_cls1 - 1) + tf.square(d_out_logits_cls1_s)) / 2
else:
    raise Exception("Unsupported number of classes.")

with tf.name_scope('d_loss'):
    if cls_num == 4:
        daf_for_global = 1.0 - ((1.0 - dynamic_adversarial_factor) / 3.0)
        d_loss = daf_for_global * d_loss_global + ((1.0 - dynamic_adversarial_factor) / 3.0) * (d_loss_cls0 + d_loss_cls1 + d_loss_cls2 + d_loss_cls3)
    elif cls_num == 2:
        daf_for_global = 1.0 - ((1.0 - dynamic_adversarial_factor) / 3.0)
        d_loss = daf_for_global * d_loss_global + ((1.0 - dynamic_adversarial_factor) / 3.0) * (d_loss_cls0 + d_loss_cls1)
    else:
        raise Exception("Unsupported number of classes.")

#生成器误差
with tf.name_scope('g_loss'):
    #对抗误差
    #g_loss_adv = tf.reduce_mean(tf.square(Dg_logits - 1)) / 2

    if cls_num == 4:
        g_loss_adv = tf.reduce_mean(tf.square(d_out_logits_global_s - 1)) / 2 + tf.reduce_mean(tf.square(d_out_logits_cls0_s - 1)) / 2 + tf.reduce_mean(tf.square(d_out_logits_cls1_s - 1)) / 2 + tf.reduce_mean(tf.square(d_out_logits_cls2_s - 1)) / 2 + tf.reduce_mean(tf.square(d_out_logits_cls3_s - 1)) / 2
    elif cls_num == 2:
        g_loss_adv = tf.reduce_mean(tf.square(d_out_logits_global_s - 1)) / 2 + tf.reduce_mean(tf.square(d_out_logits_cls0_s - 1)) / 2 + tf.reduce_mean(tf.square(d_out_logits_cls1_s - 1)) / 2



    #源领域和目标领域的分类误差
    g_loss_ce_t = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=label_layer)#+ tf.losses.get_regularization_loss()
    g_loss_ce_s = tf.losses.softmax_cross_entropy(logits=predict_s, onehot_labels=label_layer_s)


    #中心损失只用在目标域上，因为我们更关心目标域的分类，tf.argmax这里的作用是 把独热编码后的向量中最大值的下标拿出来，注意独热编码中，只有0和1。比如[1,0,0,0]就会得到0
    g_loss_center, centers = utils.get_center_loss(net2, tf.argmax(label_layer, axis=1), alpha=0.5, num_classes=cls_num, name='centers')
    #g_loss_center_s, centers_s = utils.get_center_loss(feat_s, tf.argmax(label_layer, axis=1), alpha=0.5, num_classes=cls_num, name='centers_s')

    g_loss = args.w_adv * g_loss_adv + args.w_t*g_loss_ce_t + args.w_s*g_loss_ce_s + args.w_c *g_loss_center
    loss_val = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=label_layer)#+ tf.losses.get_regularization_loss()


# split the variable for two differentiable function
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

#At each iteration, we first update the parameters of the domain discriminator, fix the feature extractor and classifier,
#and then fix the domain discriminator and update the parameters of both the feature extractor and classifier.



# optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step,
                                           10, 0.99999, staircase=True)

with tf.name_scope('train'):
    # d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*0.4).minimize(d_loss, global_step=global_step, var_list=d_vars)

    #  d_loss对d_vars中所有变量做梯度回传，并作优化
    d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, global_step=global_step, var_list=d_vars)#鉴别器优化
#    with tf.control_dependencies([centers_update_op]):
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(g_loss, var_list=g_vars)#生成器优化


## ============================ train phase ======================================
# init = tf.global_variables_initializer()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session(config = config)
saver = tf.train.Saver(max_to_keep=None)
#初始化所有变量
sess.run(init)


## ============================ save initialization ==============================

if load_model:
    # ckpt = tf.train.get_checkpoint_state('Model_and_Result/' + load_model + '/models')
    # saver.restore(sess, ckpt.model_checkpoint_path)
    ckpt = 'Model_and_Result/' + load_model + '/' + 'models/' + 'model.ckpt-0'
    saver.restore(sess, ckpt)
    print('Saved model loaded')
elif save_model:
    saver.save(sess, model_directory + '/model.ckpt-' + str(0))
    print('Begining model saved')

## ============================ start training =================================

#if (len(sourceData) % batch_size) == 0:
maxIter = int(len(sourceData) / batch_size)
#else:
    #maxIter = int(len(sourceData) / batch_size + 1)

best_loss = np.inf
stop_step = 0
early_stop_tolerance = args.stop_tolerance
best_acc = 0

trn_stime = time.time()
tst_time = 0.0
cofusion_matrix_result = None

for epoch in range(max_epoch):

    samList_s = list(range(len(sourceData)))
    samList = list(range(len(targetData)))
    # random.seed(2019)
    random.shuffle(samList_s)

    print('epoch:', epoch+1)
    lossAll, n_exp = 0, 0
    predT = []
    idxEpoch = []
    valLoss = []

    trainGLoss = 0.
    trainDloss_global = 0.
    trainDloss_local_global_advfactor = 0.
    if cls_num == 4:
        trainDloss_cls0 = 0.
        trainDloss_cls1 = 0.
        trainDloss_cls2 = 0.
        trainDloss_cls3 = 0.
    elif cls_num == 2:
        trainDloss_cls0 = 0.
        trainDloss_cls1 = 0.
    else:
        raise Exception("Unsupported number of classes.")

    for itr in range(maxIter): #这里一次循环处理一个batch的数据
        batch_trn_idx = samList_s[batch_size*itr : batch_size*(itr+1)]
        signalTrain_s = sourceData[batch_trn_idx]

        # random.shuffle(samList)
        batch_idx = np.random.choice(targetData.shape[0], len(batch_trn_idx))
        signalTrain = targetData[batch_idx]

        if len(signalTrain.shape) != 4:
            signalTrain = np.expand_dims(signalTrain, axis=-1)
            signalTrain_s = np.expand_dims(signalTrain_s, axis=-1)

        labelTrain_s = sourceLabel[batch_trn_idx]
        labelTrain = targetLabel[batch_idx]
        # labelTrain = np.expand_dims(labelTrain, axis=1)

        feed_dict = {input_layer: signalTrain, input_layer_s: signalTrain_s, label_layer: labelTrain, label_layer_s: labelTrain_s, mid_vari_DAF : DAF_flat_value}
        _, temp_DFA = sess.run([mid_assign_action, dynamic_adversarial_factor], feed_dict = feed_dict)


        #算出鉴别器损失，同时对鉴别器参数进行更新。（这两个动作谁先进行？？？？在计算图中，d_optimizer要依赖d_loss的值，所以是先计算loss再进行优化）
        if cls_num == 4:
            _, d_loss_value_global, d_loss_value_cls0, d_loss_value_cls1, d_loss_value_cls2, d_loss_value_cls3, d_loss_value_all = sess.run([d_optimizer, d_loss_global, d_loss_cls0, d_loss_cls1, d_loss_cls2, d_loss_cls3, d_loss], feed_dict=feed_dict)
        elif cls_num == 2:
            _, d_loss_value_global, d_loss_value_cls0, d_loss_value_cls1, d_loss_value_all = sess.run([d_optimizer, d_loss_global, d_loss_cls0, d_loss_cls1, d_loss], feed_dict=feed_dict)
        else:
            raise Exception("Unsupported number ofclasses.")

        # _ = sess.run([g_optimizer], feed_dict=feed_dict)
        #算出生成器损失，同时对生成器参数进行更新。（这几个动作谁先进行？？？应该是优化器最后进行），同时算出生成器的输出概率。
        _, generator_loss_value, predictV, predictV_s = sess.run([g_optimizer, g_loss, prob, prob_s], feed_dict=feed_dict)


        #if int(itr % int(maxIter / 1)) == 10000000:
        #    print('[Epoch: %2d / %2d] [%4d] loss: %.4f\n'
        #        % (epoch+1, max_epoch, itr, generator_loss_value))
        #    with open(result_dir + '/training_log.txt', 'a') as text_file:
        #        text_file.write(
        #            '[Epoch: %2d / %2d] [%4d] loss: %.4f\n'
        #            % (epoch+1, max_epoch, itr, generator_loss_value))

        trainGLoss = trainGLoss + generator_loss_value
        trainDloss_local_global_advfactor = trainDloss_local_global_advfactor + d_loss_value_all
        #trainDLoss = trainDLoss + d_loss_value * len(signalTrain)

        #trainDloss_global = trainDloss_global + d_loss_value_global * len(signalTrain)
        #trainDloss_cls0   = trainDloss_cls0 + d_loss_value_cls0 * len(signalTrain)
        #trainDloss_cls1   = trainDloss_cls1 + d_loss_value_cls1 * len(signalTrain)
        #trainDloss_cls2   = trainDloss_cls2 + d_loss_value_cls2 * len(signalTrain)
        #trainDloss_cls3   = trainDloss_cls3 + d_loss_value_cls3 * len(signalTrain)
        #每次累加一个batch的损失
        trainDloss_global = trainDloss_global + d_loss_value_global

        if cls_num == 4:
            trainDloss_cls0   = trainDloss_cls0 + d_loss_value_cls0
            trainDloss_cls1   = trainDloss_cls1 + d_loss_value_cls1
            trainDloss_cls2   = trainDloss_cls2 + d_loss_value_cls2
            trainDloss_cls3   = trainDloss_cls3 + d_loss_value_cls3
        elif cls_num == 2:
            trainDloss_cls0   = trainDloss_cls0 + d_loss_value_cls0
            trainDloss_cls1   = trainDloss_cls1 + d_loss_value_cls1
        else:
            raise Exception("Unsupported number of classes.")

        predT.extend(predictV)
        idxEpoch.extend(batch_idx)

    trainGLoss = trainGLoss/(batch_size * maxIter)
    trainDloss_global = trainDloss_global / (batch_size * maxIter)
    trainDloss_local_global_advfactor = trainDloss_local_global_advfactor / (batch_size * maxIter)

    if cls_num == 4:
        trainDloss_cls0 = trainDloss_cls0 / (batch_size * maxIter)
        trainDloss_cls1 = trainDloss_cls1 / (batch_size * maxIter)
        trainDloss_cls2 = trainDloss_cls2 / (batch_size * maxIter)
        trainDloss_cls3 = trainDloss_cls3 / (batch_size * maxIter)
    elif cls_num == 2:
        trainDloss_cls0 = trainDloss_cls0 / (batch_size * maxIter)
        trainDloss_cls1 = trainDloss_cls1 / (batch_size * maxIter)
    else:
        raise Exception("Unsupported number of classes.")

    aa = np.array(predT)
    accT = sklearn.metrics.accuracy_score(np.argmax(targetLabel[idxEpoch], 1), np.argmax(aa, 1))

    with open(result_dir + '/loss_record.txt', 'a') as loss_text_file:
        if cls_num == 4 :
            loss_text_file.write('[Epoch: %2d / %2d]  generator loss: %f, all discriminator loss: %f, global discriminator loss: %f, cls0_loss: %f, cls1_loss: %f, cls2_loss: %f, cls3_loss: %f, advf: %f\n' % (epoch+1, max_epoch,  trainGLoss, trainDloss_local_global_advfactor, trainDloss_global, trainDloss_cls0, trainDloss_cls1, trainDloss_cls2, trainDloss_cls3, DAF_flat_value))
        elif cls_num == 2:
            loss_text_file.write('[Epoch: %2d / %2d]  generator loss: %f, all discriminator loss: %f, global discriminator loss: %f, cls0_loss: %f, cls1_loss: %f, advf: %f\n' % (epoch+1, max_epoch,  trainGLoss, trainDloss_local_global_advfactor, trainDloss_global, trainDloss_cls0, trainDloss_cls1,DAF_flat_value))
        else:
            raise Exception("Unsupported number ofclasses.")

    #====================每一个epoch以后需要计算DAF，DAF初始为1.0====================================
    global_A_distance = 2.0 * (1 - 2.0 * trainDloss_global)
    if cls_num == 4:
        global_cls0_distance = 2.0 * (1 - 2.0 * trainDloss_cls0)
        global_cls1_distance = 2.0 * (1 - 2.0 * trainDloss_cls1)
        global_cls2_distance = 2.0 * (1 - 2.0 * trainDloss_cls2)
        global_cls3_distance = 2.0 * (1 - 2.0 * trainDloss_cls3)
        DAF_flat_value =  global_A_distance / (global_A_distance + ((global_cls0_distance + global_cls1_distance + global_cls2_distance + global_cls3_distance ) / cls_num))
        #DAF_flat_value =  1.0

    elif cls_num == 2:
        global_cls0_distance = 2.0 * (1 - 2.0 * trainDloss_cls0)
        global_cls1_distance = 2.0 * (1 - 2.0 * trainDloss_cls1)
        DAF_flat_value =  global_A_distance / (global_A_distance + ((global_cls0_distance + global_cls1_distance) / cls_num))
        #DAF_flat_value =  1.0
    else:
        raise Exception("Unsupported number of classes.")

    signalTest = tstData
    if len(signalTest.shape) != 4:
        signalTest = np.expand_dims(signalTest, axis=-1)

    labelTest = tstLabel
    feed_dict = {input_layer: signalTest, label_layer: labelTest}



    #测试集进行测试：
    tst_stime = time.time()
    val_loss_value, predE, feature_from_generator = sess.run([loss_val, prob, feat], feed_dict=feed_dict)  #val_loss_value定义在g_loss所属的name_space中
    tst_etime = time.time()
    tst_time += (tst_etime - tst_stime)






    acc = sklearn.metrics.accuracy_score(np.argmax(labelTest,1), np.argmax(predE,1))
    kappa = sklearn.metrics.cohen_kappa_score(np.argmax(labelTest,1), np.argmax(predE,1))

    tsne_result = TSNE(n_components=2, learning_rate=100, random_state=501, init='pca').fit_transform(feature_from_generator)  # 降至2维
    label_for_viualization = [np.argmax(i) for i in labelTest]
    fig = utils.plot_feature_map(tsne_result, label_for_viualization, 't-SNE embedding of the EEG deep features of DMDAN')


    if cls_num == 4:
        trainDLoss = (trainDloss_global + trainDloss_cls0 + trainDloss_cls1 + trainDloss_cls2 + trainDloss_cls3) / len(sourceData)
    elif cls_num == 2:
        trainDLoss = (trainDloss_global + trainDloss_cls0 + trainDloss_cls1) / len(sourceData)
    else:
        raise Exception("Unsupported number of classes.")

    print('[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'
          % (epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainGLoss, accT, trainDLoss, val_loss_value, acc, kappa))
    with open(result_dir + '/training_log.txt', 'a') as text_file:
        text_file.write("[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'\n"
                        % (epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainGLoss, accT, trainDLoss, val_loss_value, acc, kappa))


    # save model
    checkpoint_path = model_directory + '/' + 'model.ckpt'
    saver.save(sess, checkpoint_path, global_step=global_step)


    if stop_criteria == 'val_loss':
        if val_loss_value < best_loss-0.0002:
            best_loss = val_loss_value
            best_acc = acc
            best_kappa = kappa
            stop_step = 0
            best_epoch = epoch
            best_global_step = training_util.global_step(sess, global_step)
        else:
            stop_step += 1
            if stop_step > early_stop_tolerance:
                # print('Early stopping is trigger at epoch: %2d. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)'
                #       %(epoch+1, best_loss, best_acc, best_epoch+1, best_global_step))
                #
                # with open(result_dir + '/training_log.txt', 'a') as text_file:
                #     text_file.write(
                #         'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)\n'
                #         % (best_loss, best_acc, best_epoch+1, best_global_step))
                # s = open(model_directory + '/checkpoint').read()
                # s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
                # f = open(model_directory + '/checkpoint', 'w')
                # f.write(s)
                # f.close()
                break
    elif stop_criteria == 'val_acc':
        print(stop_criteria)
        if (best_acc < acc) or (abs(best_acc - acc) < 0.0001 and val_loss_value < best_loss):
            temp_label = b = np.argmax(labelTest, axis=1)
            temp_predE = b = np.argmax(predE, axis=1)

            confusion_matrix = tf.confusion_matrix(temp_label, temp_predE, cls_num)
            CM_count = confusion_matrix.eval(session=sess)
            trans_prob_mat = (CM_count.T/np.sum(CM_count, 1)).T
            #df=pd.DataFrame(trans_prob_mat, index=["leftHand", "rightHand", "feet", "tongue"], columns=["leftHand", "rightHand", "feet", "tongue"])
            df=pd.DataFrame(trans_prob_mat, index=["leftHand", "rightHand" ], columns=["leftHand", "rightHand"])
            # Plot
            plt.figure(figsize=(12.0, 12.0))
            ax = sns.heatmap(df, xticklabels=df.corr().columns,
                             yticklabels=df.corr().columns, cmap='magma',
                             linewidths=6, annot=True)
            # Decorations
            plt.xticks(fontsize=24,family='Times New Roman')
            plt.yticks(fontsize=24,family='Times New Roman')

            plt.tight_layout()
            plt.savefig(result_dir + '/confutionmatrix' + model + '.svg', transparent=True, dpi=800, format='svg')
            plt.close();



            best_acc = acc
            best_loss = val_loss_value
            best_kappa = kappa
            best_epoch = epoch
            best_global_step = training_util.global_step(sess, global_step)
            #fig_path = 'Deep_feature/fig_%d_%f.svg'% (epoch, acc)
            #fig.savefig(fig_path,format='svg', dpi=150)


trn_etime = time.time()
train_time = trn_etime - trn_stime




print('Training finished. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d) \n' % (best_loss, best_acc, best_kappa, best_epoch+1, best_global_step))

print('Training time is %.f seconds.\n' %(train_time))
print('Testing time is %.f seconds.' %    (tst_time))

with open(result_dir + '/training_log.txt', 'a') as text_file:
    text_file.write(
        'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)\n'
        % (best_loss, best_acc, best_kappa, best_epoch+1, best_global_step))

    text_file.write(
        'Training time is %.f seconds, Testing time is %.f seconds\n'
        % (train_time, tst_time))

s = open(model_directory + '/checkpoint').read()
s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
f = open(model_directory + '/checkpoint', 'w')
f.write(s)
f.close()


sess.close()
print('finished')