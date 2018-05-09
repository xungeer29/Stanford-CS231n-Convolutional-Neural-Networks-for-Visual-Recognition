# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:03:33 2018

@author: GFX
"""
from __future__ import absolute_import
import sys
sys.path.append("..")#返回上一级目录
import random
import numpy as np
from KNN.data_utils import load_cifar10
from SVM import svm_loss_naive
from SVM import svm_loss_vectorized
from gradient_check import grad_check_sparse#用于梯度检验
import time


from linear_classifer import LinearSVM
cifar10_dir='..\KNN\datasets'
x_train,y_train,x_test,y_test=load_cifar10(cifar10_dir)
#判断是否产生过拟合：从训练集中抽取一部分作为验证集
num_training=49000
num_validation=1000
num_test=1000
num_dev=500

mask=range(num_training,num_training+num_validation)#49000~50000
x_val=x_train[mask]
y_val=y_train[mask]
mask=range(num_training)#1~49000
x_train=x_train[mask]
y_train=y_train[mask]
mask=np.random.choice(num_training,num_dev,replace=False)
x_dev=x_train[mask]
y_dev=y_train[mask]
mask=range(num_test)#1~1000
x_test=x_test[mask]
y_test=y_test[mask]
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_val=np.reshape(x_val,(x_val.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
x_dev=np.reshape(x_dev,(x_dev.shape[0],-1))
mean_image=np.mean(x_train,axis=0)#axis:0 求列求平均值；1 按行求平均值
x_train-=mean_image
x_val-=mean_image
x_test-=mean_image
x_dev-=mean_image
#将 f=W*x+b 变为 f=[W,b]*X
x_train=np.hstack([x_train,np.ones((x_train.shape[0],1))])#hstack:将前后两个矩阵连接起来
x_val=np.hstack([x_val,np.ones((x_val.shape[0],1))])
x_test=np.hstack([x_test,np.ones((x_test.shape[0],1))])
x_dev=np.hstack([x_dev,np.ones((x_dev.shape[0],1))])

svm=LinearSVM()
tic=time.time()
loss_hist=svm.train(x_train,y_train,learning_rate=1e-7,reg=5e4,num_iters=1500,verbose=True)
toc=time.time()
print('that took %f s' % (toc-tic))