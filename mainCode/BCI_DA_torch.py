import numpy as np
import scipy.io as sio

import sklearn
import net
import os
import time
import random
import dataFile
import matplotlib.pyplot as plt
import shutil
import argparse
import ast
import utils
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import  keras
#from keras.utils import np_utils
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch import nn
import tensorflow.compat.v1 as tf

import DataSet
import NetWorks


## =================================== ========================================

#NumPy为了方便数据展示，采用科学计数法，但科学计数法不太便于观察，所以取消它。
np.set_printoptions(suppress=True)

#两个警告过滤器
#对于未来特性更改的警告，直接忽略
warnings.simplefilter(action='ignore', category=FutureWarning)
#对于将要过期的更改，直接忽略
warnings.simplefilter(action='ignore', category=PendingDeprecationWarning)

#参数设置
## =================================== argparse ========================================
parser = argparse.ArgumentParser()


# 两个减号代表参数名字的开始。
parser.add_argument('--model', default='test')

#选择B数据集
parser.add_argument('--dataset', default='A')

#选择第五个受害者作为目标域
parser.add_argument('--subject', default=5, type=int)

#学习率设置为0.002
parser.add_argument('--lr', default=0.002, type=float)

#一共200个epoch
parser.add_argument('--max_epoch', default=200, type=int)

#batch size为64
parser.add_argument('--batch_size', default=64, type=int)

#实验的名字叫test
parser.add_argument('--exp_name', default='test')

#没懂
parser.add_argument('--aug', default=False, type=ast.literal_eval)

#分类器选择signal_da_fc64
parser.add_argument('--classifier', default='signal_da_fc64')
#损失函数用val_loss
parser.add_argument('--criteria', default='val_loss')

#没懂
parser.add_argument('--loadN', default='1')
# parser.add_argument('--dataload', default='dataB')

#停止的限度
parser.add_argument('--stop_tolerance', default=100, type=int)
#数据长度
parser.add_argument('--data_len', default='data_0-4')

#下面4个参数没懂
parser.add_argument('--w_adv', default='0.01', type=float)
parser.add_argument('--w_t', default='1', type=float)
parser.add_argument('--w_s', default='1', type=float)
parser.add_argument('--w_c', default='0.05', type=float)

args = parser.parse_args()
#print(args)

## ===================================初始化随机种子
utils.setup_seed(100)
## =================================== path set ========================================
load_model = None  # 是否加载已有的模型，不进行模型加载
save_model = True  # 是否存储当前模型

#实验名称存储在exp_name中
exp_name = args.exp_name

try:
#如果没有在之前的参数中定义model，那么这里就会进入except分支
    model = args.model
except:
    #model = "s5" s表示subject受害者 + 5，5表示第五个
    model = 's' + str(args.subject)


#结果文件夹，存放所有结果和模型
result_dir = 'Model_and_Result' + '/' + exp_name + '/' + model
#模型文件夹，属于结果文件夹的子文件夹
model_directory = result_dir + '/models'

## data path

#dataset：A或者B，首先选择B
dataset = args.dataset
# datapath： 第几个受害者，这里是5号
dataPath = args.subject

if tf.io.gfile.exists(result_dir):
    tf.io.gfile.rmtree(result_dir)#以递归方式删除路径下的所有内容。

#也可以这么删除，就可以不用tf模块：
#if os.path.exists(result_dir):
#    shutil.rmtree

#如果文件夹不存在，则进行创建
if not os.path.exists(model_directory):
    os.makedirs(model_directory)


#把下面3个py文件和当前文件拷贝到结果文件夹中
shutil.copy(__file__, result_dir)
shutil.copy('net.py', result_dir)
shutil.copy('dataFile.py', result_dir)
shutil.copy('loadData.py', result_dir)

# os.system('cp {} {}'.format(__file__, result_dir))
# os.system('cp {} {}'.format('net.py', result_dir))

#写入代替创建，创建了一个用于训练时的log的文件
with open(result_dir + '/training_log.txt', 'w') as f:
    f.close()

## =========================== parameters set =========================

# channel_size = 240
# time_size = 150
# beta1 = 0.5
# cls_num = 2

lr = args.lr   #学习率，超参数
max_epoch = args.max_epoch #最大趟数
batch_size = args.batch_size
stop_criteria = args.criteria



# ================================   data   ==========================================================
# if dataset == 'A':
#    if dataPath < 6:
#        path = './data/dataA/0' + str(dataPath) + '/A0' + str(dataPath) + '01T.mat'
#    else:
#        path = './data/dataA/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'
# elif dataset == 'B':
#    if dataPath < 8:
#        path = './data/dataB/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
#    else:
#        path = './data/dataB/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'


if dataset == 'A':
    path = './data/BCI_IV_2a/data_0-4' + '/' + str(dataPath) + '.mat'
elif dataset == 'B':
    path = './data/BCI_IV_2b/data_0-4' + '/' + str(dataPath) + '.mat'

# trnData, trnLabel, tstData, tstLabel = dataFile.dataB_valSet(path, loadN=args.loadN, aug=args.aug)
#sourceData, sourceLabel, targetData, targetLabel, tstData, tstLabel = dataFile.data_multitask_da(aug=args.aug, dataset=dataset, loadN=args.loadN, subject=args.subject)

#数据集读入
sourceData, sourceLabel, targetData, targetLabel, tstData, tstLabel = dataFile.bciiv2a_multitask_da(subject=args.subject, data_len=args.data_len)

utils.euclidean_space_data_alignment(sourceData)



# trnData = trnData.astype('float32')
# tstData = tstData.astype('float32')

#sourceData的维度： 4608笔数据，维度22*625，来自源领域测试和训练数据的合并。
#tagetData的维度：  288笔数据，维度22*625，来自目标域的受害者训练数据。
#testData 测试数据， 288笔，维度22*625，来自目标域的受害者测试数据。


#暂时不用data set 类
#sourceSet = DataSet.BCI_Dataset(data = sourceData, label = sourceLabel);
#targetSet = DataSet.BCI_Dataset(data = targetData, label = targetLabel);
#tstSet    = DataSet.BCI_Dataset(data = tstData,    label = tstLabel);

#source_set_itr = DataLoader(dataset=sourceSet,batch_size=batch_size,shuffle=True,drop_last=False);
#target_set_itr = DataLoader(dataset=targetSet,batch_size=batch_size,shuffle=True,drop_last=False);
#tst_set_itr    = DataLoader(dataset=tstSet,   batch_size=batch_size,shuffle=True,drop_last=False);



#打印标签种类，并且统计各个标签对应的数目
print('Source set label and proportion:\t', np.unique(sourceLabel, return_counts=True))
print('Target set label and proportion:\t', np.unique(targetLabel, return_counts=True))
print(' Val   set label and proportion:\t', np.unique(tstLabel, return_counts=True)) #验证或者测试集


#类别数目：4类
cls_num = len(np.unique(targetLabel))
dataSize = targetData.shape
#数据的维度是：(288, 22, 625)22代表22个头部采样点，625是实验时间点，288样本数

#22个通道，每个通道分别为625的时间序列
#EEGnet
channel_size = dataSize[1] #22
time_size = dataSize[2]    #625
try:
    depth_size = dataSize[3]#没有第三个维度，仅仅进行MI实验
except:
    depth_size = 1



#转换成one-hot  vector
#0 -> [1,0,0,0]
#1 -> [0,1,0,0]
#2 -> [0,0,1,0]
#3 -> [0,0,0,1]
targetLabel = keras.utils.np_utils.to_categorical(targetLabel, num_classes=cls_num)
sourceLabel = keras.utils.np_utils.to_categorical(sourceLabel, num_classes=cls_num)
tstLabel    = keras.utils.np_utils.to_categorical(tstLabel   , num_classes=cls_num)




## ============================= model ===============================
#决定生成器和辨别器

#生成器  传入的参数                   1  22 625 4
generator = NetWorks.GeneratorNet(depth_size, channel_size, time_size, cls_num);
#鉴别器
discriminator = NetWorks.DiscriminatorNet(64);



## ============================ save initialization ==============================
if load_model:
# ckpt = tf.train.get_checkpoint_state('Model_and_Result/' + load_model + '/models')
# saver.restore(sess, ckpt.model_checkpoint_path)
#ckpt = 'Model_and_Result/' + load_model + '/' + 'models/' + 'model.ckpt-0'
#saver.restore(sess, ckpt)
#print('Saved model loaded')
    print('Model loading not yet implemented.');
elif save_model:
#saver.save(sess, model_directory + '/model.ckpt-' + str(0))
#print('Begining model saved')
    print('Model saving not yet implemented.')

#训练
## ============================ start training =================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if (len(sourceData) % batch_size) == 0:
    maxIter = int(len(sourceData) / batch_size)
else:
    #最后一个batch大小不足batch_size的情况
    maxIter = int(len(sourceData) / batch_size + 1)

best_loss = np.inf #最佳损失
stop_step = 0
early_stop_tolerance = args.stop_tolerance
best_acc = 0 #最佳准确率

#把模型和数据送入gpu
generator = generator.to(device)
discriminator = discriminator.to(device)


#训练开始时间
stime = time.time()

#定义优化器
d_opti      = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False);
g_opti      = optim.Adam(generator.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False);

#定义 learning rate
#TODO:学习率，没有tf里面exponential_decay那样精准的控制，该怎么办？
#g_scheduler = ExponentialLR(d_opti, gamma=0.99999)
#d_scheduler = ExponentialLR(g_opti, gamma=0.99999)

for epoch in range(max_epoch):

    #源领域
    #[0,1,2,3， ... len(sourceData)-1]
    samList_s = list(range(len(sourceData)))

    #目标领域
    #[0,1,2,3， ... len(targetData)-1]
    samList = list(range(len(targetData)))

    random.shuffle(samList_s)
    print('epoch:', epoch + 1)

    trainLoss = 0
    valLoss = []
    lossAll, n_exp = 0, 0
    predT = []
    idxEpoch = []
    trainDLoss = 0

    for itr in range(maxIter):
        #-------------------------------------------数据获取------------------------------------------------
        #一个batch中会用到的下标的数组
        batch_trn_idx = samList_s[batch_size * itr: batch_size * (itr + 1)]
        # random.shuffle(samList)

        #拿出一个batch的source的数据
        signalTrain_s = sourceData[batch_trn_idx]

        #拿出一个batch的target的数据
        batch_idx = np.random.choice(targetData.shape[0], len(batch_trn_idx))
        signalTrain = targetData[batch_idx]

        #扩展成4维，第二维度就是通道数目，这里为1
        if len(signalTrain.shape) != 4:
            temp = np.expand_dims(signalTrain, axis=1)
            signalTrain = np.expand_dims(signalTrain, axis=1)
            signalTrain_s = np.expand_dims(signalTrain_s, axis=1)
        #标签（已经是onehot编码之后的）
        labelTrain = targetLabel[batch_idx]
        labelTrain_s = sourceLabel[batch_trn_idx]

        #-------------------------------------------TODO: 使用dynamic adoptation-------------------------------------------
        #p = float(batch_idx+1 + epoch * len_dataloader) / args.epochs / len_dataloader
        #alpha = 2. / (1. + np.exp(-10 * p)) - 1

        #optimizer.zero_grad()
        #source_data, source_label = source_data.to(device), source_label.to(device)

        #source_data, source_label = signalTrain_s.to(device), labelTrain_s.to(device)
        #for target_data, target_label in target_loader:
        #target_data, target_label = signalTrain.to(device), labelTrain.to(device)


        #out = model(source_data, target_data, source_label, DEV, alpha)
        #s_output, s_domain_output, t_domain_output = out[0],out[1],out[2]
        #s_out = out[3]
        #t_out = out[4]

        ##global loss
        #sdomain_label = torch.zeros(args.batch_size).long().to(DEV)
        #err_s_domain = F.nll_loss(F.log_softmax(s_domain_output, dim=1), sdomain_label)
        #tdomain_label = torch.ones(args.batch_size).long().to(DEV)
        #err_t_domain = F.nll_loss(F.log_softmax(t_domain_output, dim=1), tdomain_label)

        ##local loss
        #loss_s = 0.0
        #loss_t = 0.0
        #tmpd_c = 0
        #for i in range(args.num_class):
        #    loss_si = F.nll_loss(F.log_softmax(s_out[i], dim=1), sdomain_label)
        #    loss_ti = F.nll_loss(F.log_softmax(t_out[i], dim=1), tdomain_label)
        #    loss_s += loss_si
        #    loss_t += loss_ti
        #    tmpd_c += 2 * (1 - 2 * (loss_si + loss_ti))
        #tmpd_c /= args.num_class

        #d_c = d_c + tmpd_c.cpu().item()

        #global_loss = 0.05*(err_s_domain + err_t_domain)
        #local_loss = 0.01*(loss_s + loss_t)

        #d_m = d_m + 2 * (1 - 2 * global_loss.cpu().item())

        #join_loss = (1 - MU) * global_loss + MU * local_loss
        #soft_loss = F.nll_loss(F.log_softmax(s_output, dim=1), source_label)
        #if args.gamma == 1:
        #    gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        #if args.gamma == 2:
        #    gamma = epoch /args.epochs
        #loss = soft_loss + join_loss
        #loss.backward()
        #optimizer.step()

        #if batch_idx % args.log_interval == 0:
        #    print('\nLoss: {:.6f},  label_Loss: {:.6f},  join_Loss: {:.6f}, global_Loss:{:.4f}, local_Loss:{:.4f}'.format(
        #        loss.item(), soft_loss.item(), join_loss.item(), global_loss.item(), local_loss.item()))
        ##total_progress_bar.update(1)
        #D_M = np.copy(d_m).item()
        #D_C = np.copy(d_c).item()



        #------------------------------------------生成器前向传播---------------------------------------------------
        #TODO:网络采用多输出时用hook和直接输出的区别？
        predict, prob, feat, net2         = generator.forward(signalTrain)
        predict_s, prob_s, feat_s, net2_s = generator.forward(signalTrain_s)
        #------------------------------------------鉴别器前向传播---------------------------------------------------
        Dx, Dx_logits = discriminator(feat)     #目标领域
        Dg, Dg_logits = discriminator(feat_s)  # fake，源领域


        #TODO:更加agressive的梯度优化
        #-------------------------------------------对鉴别器进行训练------------------------------------------------
        d_loss = torch.mean(torch.square(Dx_logits - 1) + torch.square(Dg_logits)) / 2
        #首先 清零 鉴别器中所有参数的梯度。
        d_opti.zero_grad()
        #后向传播梯度
        d_loss.backward()
        #更新参数
        d_opti.step()
        #更新学习率
        #d_scheduler.step()
        #-------------------------------------------对生成器进行训练------------------------------------------------
        #损失函数计算
        g_loss_adv = torch.mean(torch.square(Dg_logits - 1)) / 2
        logits_predict  = F.softmax(predict)
        g_loss_ce_t = F.cross_entropy(logits_predict,labelTrain)  #源领域和样本标签的交叉熵

        logits_predict_s  = F.softmax(predict_s)
        g_loss_ce_s = F.cross_entropy(logits_predict_s,labelTrain_s)   #目标领域和样本标签的交叉熵

        #TODO:center loss计算方式变成torch版本
        #计算center loss
        g_loss_center, centers = utils.get_center_loss_torch(net2, tf.argmax(label_layer, axis=1), alpha=0.5, num_classes=cls_num)
        #g_loss_center_s, centers_s = utils.get_center_loss(feat_s, tf.argmax(label_layer, axis=1), alpha=0.5, num_classes=cls_num, name='centers_s')

        g_loss = args.w_adv * g_loss_adv + args.w_t * g_loss_ce_t + args.w_s * g_loss_ce_s + args.w_c * g_loss_center

        #清零生成器参数当中的所有梯度
        g_opti.zero_grad()
        #反向传播参数的梯度
        g_loss.backward()
        #更新参数
        g_opti.step()
        #更新学习率
        g_scheduler.step()



        if int(itr % int(maxIter / 1)) == 10000000:
            print('[Epoch: %2d / %2d] [%4d] loss: %.4f\n'% (epoch + 1, max_epoch, itr, g_loss))
        with open(result_dir + '/training_log.txt', 'a') as text_file:
                text_file.write('[Epoch: %2d / %2d] [%4d] loss: %.4f\n'% (epoch + 1, max_epoch, itr, g_loss))

        trainLoss = trainLoss + g_loss * len(signalTrain)
        predT.extend(prob)
        idxEpoch.extend(batch_idx)
        trainDLoss = trainDLoss + d_loss * len(signalTrain)

    trainLoss = trainLoss / len(sourceData)
    aa = np.array(predT)
    accT = sklearn.metrics.accuracy_score(np.argmax(targetLabel[idxEpoch], 1), np.argmax(aa, 1))
    trainDLoss = trainDLoss / len(sourceData)

    signalTest = tstData
    #扩展第二维度，使之称为1维，因为torch和tensorflow接受输入的方式不一样
    if len(signalTest.shape) != 4:
        signalTest = np.expand_dims(signalTest, axis=1)

    labelTest = tstLabel
    #feed_dict = {input_layer: signalTest, label_layer: labelTest}

    #-------------------------------------------测试集上的损失计算，已经预测结果------------------------------------------------
    predict, prob, feat, net2         = generator.forward(tstData)
    logits_predict  = F.softmax(predict)
    predE = logits_predict
    val_loss_value = F.cross_entropy(logits_predict,tstLabel)



    acc = sklearn.metrics.accuracy_score(np.argmax(labelTest, 1), np.argmax(predE, 1))
    kappa = sklearn.metrics.cohen_kappa_score(np.argmax(labelTest, 1), np.argmax(predE, 1))

    print(
        '[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'
        % (
            epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainLoss, accT, trainDLoss, val_loss_value,
            acc, kappa))
    with open(result_dir + '/training_log.txt', 'a') as text_file:
        text_file.write(
            "[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'\n"
            % (epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainLoss, accT, trainDLoss,
               val_loss_value, acc, kappa))

    # save model
    checkpoint_path = model_directory + '/' + 'model.ckpt'
    saver.save(sess, checkpoint_path, global_step=global_step)

    if stop_criteria == 'val_loss':
        if val_loss_value < best_loss - 0.0002:
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
        if (best_acc < acc) or (abs(best_acc - acc) < 0.0001 and val_loss_value < best_loss):
            best_acc = acc
            best_loss = val_loss_value
            best_kappa = kappa
            best_epoch = epoch
            best_global_step = training_util.global_step(sess, global_step)

print('Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)'
      % (best_loss, best_acc, best_kappa, best_epoch + 1, best_global_step))
with open(result_dir + '/training_log.txt', 'a') as text_file:
    text_file.write(
        'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)\n'
        % (best_loss, best_acc, best_kappa, best_epoch + 1, best_global_step))
s = open(model_directory + '/checkpoint').read()
s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"',
              'model_checkpoint_path: "model.ckpt-' + str(best_global_step) + '"')
f = open(model_directory + '/checkpoint', 'w')
f.write(s)
f.close()


#进行验证
with torch.no_grad(): #不进行任何梯度的传播
    #------------------------------------------生成器前向传播---------------------------------------------------
    #TODO:多输出用hook和直接输出的区别？
    predict_test, prob_test, feat_test, net2_test         = generator.forward(signalTest)

    logits_predict_test  = F.softmax(predict_test)
    val_loss_value = F.cross_entropy(logits_predict_test,labelTest)
    predE = prob_test
    #val_loss_value, predE = sess.run([loss_val, prob], feed_dict=feed_dict)

    acc = sklearn.metrics.accuracy_score(np.argmax(labelTest, 1), np.argmax(predE, 1))
    kappa = sklearn.metrics.cohen_kappa_score(np.argmax(labelTest, 1), np.argmax(predE, 1))

    print(
        '[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'
        % (
            epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainLoss, accT, trainDLoss, val_loss_value,
            acc, kappa))
    with open(result_dir + '/training_log.txt', 'a') as text_file:
        text_file.write(
            "[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'\n"
            % (epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainLoss, accT, trainDLoss,
               val_loss_value, acc, kappa))




print('Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)'
      % (best_loss, best_acc, best_kappa, best_epoch + 1, best_global_step))
with open(result_dir + '/training_log.txt', 'a') as text_file:
    text_file.write(
        'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)\n'
        % (best_loss, best_acc, best_kappa, best_epoch + 1, best_global_step))
s = open(model_directory + '/checkpoint').read()
s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"',
              'model_checkpoint_path: "model.ckpt-' + str(best_global_step) + '"')
f = open(model_directory + '/checkpoint', 'w')
f.write(s)
f.close()

print('finished')


#========================================================模型存储========================================================
#    # save model
#    checkpoint_path = model_directory + '/' + 'model.ckpt'
#    saver.save(sess, checkpoint_path, global_step=global_step)
#
#    if stop_criteria == 'val_loss':
#        if val_loss_value < best_loss - 0.0002:
#            best_loss = val_loss_value
#            best_acc = acc
#            best_kappa = kappa
#            stop_step = 0
#            best_epoch = epoch
#            best_global_step = training_util.global_step(sess, global_step)
#        else:
#            stop_step += 1
#            if stop_step > early_stop_tolerance:
#                 print('Early stopping is trigger at epoch: %2d. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)'
#                       %(epoch+1, best_loss, best_acc, best_epoch+1, best_global_step))
#
#                 with open(result_dir + '/training_log.txt', 'a') as text_file:
#                     text_file.write(
#                         'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)\n'
#                         % (best_loss, best_acc, best_epoch+1, best_global_step))
#                 s = open(model_directory + '/checkpoint').read()
#                 s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
#                 f = open(model_directory + '/checkpoint', 'w')
#                 f.write(s)
#                 f.close()
#                 #break
#    elif stop_criteria == 'val_acc':
#        if (best_acc < acc) or (abs(best_acc - acc) < 0.0001 and val_loss_value < best_loss):
#            best_acc = acc
#            best_loss = val_loss_value
#            best_kappa = kappa
#            best_epoch = epoch
#            best_global_step = training_util.global_step(sess, global_step)
#
#print('Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)'
#      % (best_loss, best_acc, best_kappa, best_epoch + 1, best_global_step))
#with open(result_dir + '/training_log.txt', 'a') as text_file:
#    text_file.write(
#        'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)\n'
#        % (best_loss, best_acc, best_kappa, best_epoch + 1, best_global_step))
#s = open(model_directory + '/checkpoint').read()
#s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"',
#              'model_checkpoint_path: "model.ckpt-' + str(best_global_step) + '"')
#f = open(model_directory + '/checkpoint', 'w')
#f.write(s)
#f.close()
#
#sess.close()
#print('finished')











