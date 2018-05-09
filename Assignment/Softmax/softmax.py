# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:43:28 2018

@author: GFX
"""

# Softmax
# 1、模型构建
# 2、数据处理
# 3、梯度计算和检验
# 4、训练和预测
# 5、权重可视化

# 1、Softmax线性分类器构建
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)#数值稳定，平移到最大值为0
        loss_i = shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))#交叉熵损失 公式
        loss += loss + i
        for j in range(num_classes):
            softmax_out = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))#softmax函数
            if j == y[i]:#？？？？？？？？？？？
                dW[:, j] += (-1 + softmax_out) * X[i]#？？？？？？？？
            else:
                dW[:, j] += softmax_out * X[i]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0;
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)

    shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)  # 先转成(N,1)

    softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape((-1, 1))#softmax损失函数向量化

    loss = -np.sum(
        np.log(softmax_output[range(num_train), list(y)]))  # softmax_output[range(num_train),list(y)]计算正确分类y_i的损失
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dS = softmax_output.copy()
    dS[range(num_train), list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW / num_train + reg * W
    return loss, dW

##import numpy as np
# import sys
# sys.path.append("..")
##from softmax import *
#
# class LinearClassifier:
#    def __init__(self):
#        self.W=None
#        
#    def train(self,X,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
#        num_train,dim=X.shape
#        num_classes=np.max(y)+1 #
#        if self.W is None:
#            # lazily initialize W
#            self.W=0.001*np.random.randn(dim,num_classes)
#            # Run stochastic gradient descent to optimize W
#            loss_history=[]
#        for it in range(num_iters):
#            X_batch=None
#            y_batch=None
#                
#            batch_idx=np.random.choice(num_train,batch_size,replace=True)
#            X_batch=X[batch_idx]
#            y_batch=y[batch_idx]
#            
#            #
#            loss,grad=self.loss(X_batch,y_batch,reg)
#            
#            loss_history.append(loss)
#            
#            self.W+=-1*learning_rate*grad
#            if verbose and it%100==0:
#                print('iteration %d /%d:loss %f' % (it,num_iters,loss))
#                
#        return loss_history
#    def predict(self,X):
#        y_pred=np.zeros(X.shape[1])
#        scores=X.dot(self.W)
#        y_pred=np.argmax(scores,axis=1)
#        return y_pred
#    
#    def loss(self,X_batch,y_batch,reg):
#        pass
#    
# class Softmax(LinearClassifier):
#    
#    def loss(self,X_batch,y_batch,reg):
#        return softmax_loss_vectorized(self.W,X_batch,y_batch,reg)
