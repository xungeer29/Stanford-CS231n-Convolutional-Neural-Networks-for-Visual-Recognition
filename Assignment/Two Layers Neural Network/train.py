#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/15 13:19
# @Author  : GFX
# @Site    : 
# @File    : train.py
# @Software: PyCharm

# 3 模型训练
# 3.1 网络训练初始化

from neural_net import TwoLayerNet
from DataPreprocess import *
# =============================================================================
# from numpy import mean
# =============================================================================

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)
stats = net.train(X_train, y_train, X_val, y_val, num_iters=1000, batch_size=200,
              learning_rate=1e-4, learning_rate_decay=0.95, reg=0.5, verbose=True)
val_acc = (net.predict(X_val) == y_val).mean()
print('valiadation accuracy:', val_acc)
# 输出：
# iteration 0 / 1000 : loss 2.302975
# iteration 100 / 1000 : loss 2.302409
# iteration 200 / 1000 : loss 2.297453
# iteration 300 / 1000 : loss 2.274700
# iteration 400 / 1000 : loss 2.211710
# iteration 500 / 1000 : loss 2.126385
# iteration 600 / 1000 : loss 2.074668
# iteration 700 / 1000 : loss 2.056960
# iteration 800 / 1000 : loss 2.002378
# iteration 900 / 1000 : loss 2.004737
# valiadation accuracy: 0.279


# =============================================================================
# import matplotlib.pyplot as plt
# 
# 
# plt.plot()
# 
# plt.subplot(211)
# plt.plot(stats['loss_history'])
# plt.title('loss history')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# 
# plt.subplot(212)
# plt.plot(stats['train_acc_history'],label='train')
# plt.plot(stats['val_acc_history'],label='val')
# plt.title('classification accuracy history')
# plt.xlabel('epoch')
# plt.ylabel('classification accuracy')
# plt.show()
# =============================================================================
