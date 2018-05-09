# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:59:26 2018

@author: GFX
"""

# 2 数据处理
from __future__ import absolute_import
import sys
# sys.path.append("..")#返回上一级目录
import random
import numpy as np
from data_utils import load_cifar10
from SVM import svm_loss_naive
from SVM import svm_loss_vectorized
from gradient_check import grad_check_sparse  # 用于梯度检验
import time

cifar10_dir = 'datasets'
x_train, y_train, x_test, y_test = load_cifar10(cifar10_dir)
# 验证结果是否正确
print('\n验证结果是否正确')
print('training data shape: ', x_train.shape)
print('training labels shape: ', y_train.shape)
print('test data shape: ', x_test.shape)
print('test labels shape: ', y_test.shape)

# 判断是否产生过拟合：从训练集中抽取一部分作为验证集
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = range(num_training, num_training + num_validation)  # 49000~50000
x_val = x_train[mask]
y_val = y_train[mask]
mask = range(num_training)  # 1~49000
x_train = x_train[mask]
y_train = y_train[mask]
mask = np.random.choice(num_training, num_dev, replace=False)
x_dev = x_train[mask]
y_dev = y_train[mask]
mask = range(num_test)  # 1~1000
x_test = x_test[mask]
y_test = y_test[mask]
# 验证结果是否正确
print('\n验证分离验证集结果是否正确')
print('training data shape: ', x_train.shape)
print('training labels shape: ', y_train.shape)
print('validation data shape: ', x_val.shape)
print('validation data shape: ', y_val.shape)
print('test data shape: ', x_test.shape)
print('test labels shape: ', y_test.shape)

# 将图像的三维数组拉成一维数组
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_val = np.reshape(x_val, (x_val.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_dev = np.reshape(x_dev, (x_dev.shape[0], -1))
print('\n验证三维到一维的转换结果是否正确')
print('training data shape: ', x_train.shape)
# print('training labels shape: ',y_train.shape)
print('validation data shape: ', x_val.shape)
# print('validation data shape: ',y_val.shape)
print('test data shape: ', x_test.shape)
# print('test labels shape: ',y_test.shape)
print('development data shape: ', x_dev.shape)

# 数据归一化处理
# 处理方法：对每特征值减去平均值来中心化
mean_image = np.mean(x_train, axis=0)  # axis:0 求列求平均值；1 按行求平均值
x_train -= mean_image
x_val -= mean_image
x_test -= mean_image
x_dev -= mean_image
# 将 f=W*x+b 变为 f=[W,b]*X
x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])  # hstack:将前后两个矩阵连接起来
x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
x_dev = np.hstack([x_dev, np.ones((x_dev.shape[0], 1))])
# 输出检查
print('\n验证将b加入到W中的转换结果是否正确')
print('training data shape: ', x_train.shape)
print('validation data shape: ', x_val.shape)
print('test data shape: ', x_test.shape)
print('development data shape: ', x_dev.shape)

# 3 梯度计算
# 数值计算方式
w = np.random.randn(3073, 10) * 0.0001
loss, grad = svm_loss_naive(w, x_dev, y_dev, 0.00001)
print('loss(数值计算) is : %f \n' % loss)  # 9.651409 因为随机产生参数，每次运行结果会略有变化
# 梯度检验
# 将数值梯度法与分析梯度法计算结果进行比较
loss, grad = svm_loss_naive(w, x_dev, y_dev, 1e2)
f = lambda w: svm_loss_naive(w, x_dev, y_dev, 1e2)[0]
grad_numberical = grad_check_sparse(f, w, grad)

# 两种方式比较
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(w, x_dev, y_dev, 0.00001)
toc = time.time()
print('\n naive loss: %e computed in %f s' % (loss_naive, toc - tic))
tic = time.time()
loss_vectorized, grad_vectorized = svm_loss_vectorized(w, x_dev, y_dev, 0.00001)
toc = time.time()
print('vectorized loss: %e computed in %f s' % (loss_vectorized, toc - tic))
print('difference: %e \n' % (loss_naive - loss_vectorized))

# 4 模型训练
# 4.1 梯度更新
from linear_classifer import LinearSVM

svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(x_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
toc = time.time()
print('that took %f s' % (toc - tic))

# 保存参数，使用参数进行预测，计算准确率
print('\n 保存参数，使用参数进行预测，计算准确率')
y_train_pred = svm.predict(x_train)
print('training accuracy: %f ' % (np.mean(y_train == y_train_pred)))  # 计算正确预测分类的准确率
y_val_pred = svm.predict(x_val)
print('validation accuracy: %f ' % (np.mean(y_val == y_val_pred)))  # 验证集

# 超参数(学习速率和正则项)调优
# 通过交叉验证来选择较好的学习率和正则项系数
print('\n 超参数(学习速率和正则项)调优')
learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
regularization_strengths = [(1 + i * 0.1) * 1e4 for i in range(-3, 3)] + [(2 + 0.1 * i) * 1e4 for i in range(-3, 3)]
results = {}
best_val = -1
best_svm = None
for learning in learning_rates:
    for regularization in regularization_strengths:
        svm = LinearSVM()
        svm.train(x_train, y_train, learning_rate=learning, reg=regularization, num_iters=2000)
        y_train_pred = svm.predict(x_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        print('training accuracy: %f' % (train_accuracy))
        y_val_pred = svm.predict(x_val)
        val_accuracy = np.mean(y_val == y_val_pred)
        print('validation accuracy: %f' % (val_accuracy))
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm
        results[(learning, regularization)] = (train_accuracy, val_accuracy)
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross‐validation: %f' % best_val)
# 以图形的形式将上述结果可视化
# 面积大小表示正确率
import math
import matplotlib.pyplot as plt

x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

sz = [results[x][0] * 1500 for x in results]
plt.subplot(1, 2, 1)
plt.scatter(x_scatter, y_scatter, sz)
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('cifar10 training accuracy')

sz = [results[x][1] * 1500 for x in results]
plt.subplot(1, 2, 2)
plt.scatter(x_scatter, y_scatter, sz)
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('cifar10 validation accuracy')
plt.show()

# 5 模型预测
y_test_pred = best_svm.predict(x_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear svm on raw pixels final test set accuracy: %f' % test_accuracy)

# 6 权重可视化
w = best_svm.W[:-1, :]
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
