#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 20:41
# @Author  : GFX
# @Site    :
# @File    : Train_and_Predict.py
# @Software: PyCharm

#1 两层神经网络
#1.1 损失函数和梯度
import numpy as np

class TwoLayerNet(object):

    def __init__(self,input_size,hidden_size,output_size,std=1e-4):
        self.params={}
        self.params['W1']=std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
    
    # X (N,D)维输入数据  y (N,)维标签
    # W1(D,H)第一层权重 b1(H,)第一层偏置
    # W2(H,C)第二层权重 b1(C,)第二层偏置
    def loss(self,X,y=None,reg=0.0):
        W1,b1=self.params['W1'],self.params['b1']#损失函数
        W2,b2=self.params['W2'],self.params['b2']
        N,D=X.shape
    
        scores=None
    
        #激活函数
        h_output = np.maximum(0, X.dot(W1) + b1)  # 第一层输出(N,H)，Relu激活函数f(x)=max(0, x)
        #第一层神经网络的输出
        scores = h_output.dot(W2) + b2  # 第二层激活函数前的输出(N,C)
    
        if y is None:
            return scores
        loss = None
    
        #对scores进行归一化（所有scores减去各自行的最大值）
        shift_scores = scores-np.max(scores, axis=1).reshape((-1, 1))
    
        #第二层的激活函数softmax：f(x)=e^f / sum(e^f)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        #交叉熵损失
        loss =-np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  # L2正则项，防止过拟合 1/2 * lambda * w^2
    
        grads = {}
    
    
        # 第二层梯度计算
        dscores = softmax_output.copy()
        dscores[range(N), list(y)]-=1
        dscores /= N
        grads['W2'] = h_output.T.dot(dscores) + reg * W2    #梯度更新 W2
        grads['b2'] = np.sum(dscores, axis=0)   #参数b更新
    
        # 第一层梯度计算
        dh = dscores.dot(W2.T)
        dh_ReLu = (h_output > 0) * dh   #max门最大值直接获取上游梯度
        grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
        grads['b1'] = np.sum(dh_ReLu, axis=0)
    
        return loss, grads
    
    
    # 1.2 训练
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)  # 每一轮迭代数目
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
    
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
    
            # 参数更新
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
    
            if verbose and it % 100 == 0:  # 每迭代100次，打印
                print('iteration %d / %d : loss %f' % (it, num_iters, loss))
    
            if it % iterations_per_epoch == 0:  # 一轮迭代结束
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                # 更新学习率
                learning_rate *= learning_rate_decay
                
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }
    
    
    # 1.3 预测
    def predict(self, X):
        y_pred = None
        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        
        return y_pred
