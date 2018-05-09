# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:17:11 2018

@author: GFX
"""

#训练和预测

#将数据集载入模型
import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
from KNN import KNearestNeighbor
import time

x_train,y_train,x_test,y_test=load_cifar10('datasets')

#验证结果是否正确
print('\n验证结果是否正确')
print('training data shape: ',x_train.shape)
print('training labels shape: ',y_train.shape)
print('test data shape: ',x_test.shape)
print('test labels shape: ',y_test.shape)

#从这50000张训练集每一类中随机挑选 samples_per_class 张图片进行展示
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes=len(classes)
samples_per_class=7#样本数量
for y,cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y)#flatnonzero：返回扁平化后矩阵中非零元素的位置（index），也就是标签为1(属于该类)的图像的位置
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    for i,idx in enumerate(idxs):
        plt_idx=i*num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i==0:
            plt.title(cls)
plt.show()

#选取5000张训练集，500张测试集图像进行训练
num_training=5000
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
num_test=500
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]

#为了方便计算欧氏距离，将图像数据拉长成行向量
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
print(x_train.shape,x_test.shape)#输出选取的训练集和测试集的大小

#测试集预测
#计算每个测试集图像与训练集图像的欧氏距离 
classifier=KNearestNeighbor()
classifier.train(x_train,y_train)
dists_two=classifier.compute_distances_two_loops(x_test)#采用distances_two_loops方式计算欧式距离
print(dists_two)#输出计算的欧式距离
#预测测试集类别
y_test_pred=classifier.predict_labels(dists_two,k=1)

#准确率计算
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print ('\n Got %d / %d correct =>accuracy: %f' % (num_correct,num_test,accuracy))

#使用compute_distances_one_loops方式计算欧式距离，输出与two_loops的差别
dists_one=classifier.compute_distances_one_loops(x_test)
difference=np.linalg.norm(dists_two - dists_one,ord='fro')
print('two-one difference was: %f' %difference)

#使用compute_distances_no_loops方式计算欧式距离，输出与two_loops的差别
dists_no=classifier.compute_distances_no_loops(x_test)
difference=np.linalg.norm(dists_two - dists_no,ord='fro')
print('two-no difference was: %f' % difference)

#比较三种计算欧式距离的花费时间
#no_loops的花费时间最小
def time_function(f,*args):
    tic=time.time()
    f(*args)
    toc=time.time()
    return toc-tic
two_loop_time=time_function(classifier.compute_distances_two_loops,x_test)
print('two loops version took %f seconds' % two_loop_time)
one_loop_time=time_function(classifier.compute_distances_one_loop,x_test)
print('one loop version took %f seconds' % one_loop_time)
no_loops_time=time_function(classifier.compute_distances_no_loops,x_test)
print('no loops version took %f seconds' % no_loops_time)

#交叉验证
#利用交叉验证来选择最好的 k 值来获得较好的预测的准确率
num_folds=5 #将数据平分成5份
k_choices=[1,3,5,8,10,12,15,20,50,100]
x_train_folds=[]
y_train_folds=[]
y_train=y_train.reshape(-1,1)
x_train_folds=np.array_split(x_train,num_folds)  #数据集平分成5份
y_train_folds=np.array_split(y_train,num_folds) #
k_to_accuracies={}  #以字典形式存储 k 和 accuracy
for k in k_choices:
    k_to_accuracies.setdefault(k,[])
for i in range(num_folds):    #对每个 k 值，选取一份测试，其余训练，计算准确率
    classifier=KNearestNeighbor()
    x_val_train=np.vstack(x_train_folds[0:i]+x_train_folds[i+1:])    #除i之外的作为训练集
    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i + 1:])
    y_val_train=y_val_train[:,0]
    classifier.train(x_val_train,y_val_train)
    for k in k_choices:
        y_val_pred=classifier.predict(x_train_folds[i],k=k)    #第i份作为测试集并预测
        num_correct=np.sum(y_val_pred==y_train_folds[i][:,0])
        accuracy=float(num_correct)/len(y_val_pred)
        k_to_accuracies[k]=k_to_accuracies[k]+[accuracy]
for k in sorted(k_to_accuracies):    #表示输出每次得到的准确率以及每个k值对应的平均准确率
    sum_accuracy=0
    for accuracy in k_to_accuracies[k]:
        print('k=%d, accuracy=%f' % (k,accuracy))
        sum_accuracy+=accuracy
    print('the average accuracy is :%f' % (sum_accuracy/5))
    
#计算均值，画出曲线
for k in k_choices:
    accuracies=k_to_accuracies[k]
    plt.scatter([k]*len(accuracies),accuracies)
accuracies_mean=np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std=np.array([np.std(v) for k ,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices,accuracies_mean,yerr=accuracies_std)
plt.title('cross‐validation on k')
plt.xlabel('k')
plt.ylabel('cross‐validation accuracy')
plt.show()

#使用k=10来完成预测任务
best_k=10
classifier=KNearestNeighbor()
classifier.train(x_train,y_train)
y_test_pred=classifier.predict(x_test,k=best_k)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('got %d / %d correct => accuracy: %f' %(num_correct,num_test,accuracy))




