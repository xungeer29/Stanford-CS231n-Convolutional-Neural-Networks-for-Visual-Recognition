#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 20:41
# @Author  : GFX
# @Site    : 
# @File    : Train_and_Predict.py
# @Software: PyCharm

# 2 数据处理 图像转化为数组，归一化处理(减去均值)

import numpy as np
from data_utils import load_cifar10


def get_cifar_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = 'datasets'
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)
    # 验证集
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    # 训练集
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    # 测试集
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 数据归一化处理
    # 处理方法：对每特征值减去平均值来中心化
    mean_image = np.mean(X_train, axis=0)   # axis:0 求列求平均值；1 按行求平均值
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    #将图像转化为列向量
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# 验证结果是否正确
X_train, y_train, X_val, y_val, X_test, y_test = get_cifar_data()
print('\n验证分离验证集结果是否正确')
print('training data shape: ', X_train.shape)
print('training labels shape: ', y_train.shape)
print('validation data shape: ', X_val.shape)
print('validation data shape: ', y_val.shape)
print('test data shape: ', X_test.shape)
print('test labels shape: ', y_test.shape)
