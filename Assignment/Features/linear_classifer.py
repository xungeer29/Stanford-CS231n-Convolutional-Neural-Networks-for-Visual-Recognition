# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:46:43 2018

@author: GFX
"""

#1.2 分类器的构建

#构建训练和预测的模型
import numpy as np
from SVM import *
import matplotlib.pyplot as plt

class LinearClassifier:
    def __init__(self):
        self.W=None
    #训练模型  
    #采用随机梯度法进行训练（SGD）
    #batch_size批大小
    #verbose 为True时显示中间迭代过程，输出是每次迭代的损失函数值
    def train(self,X,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train,dim=X.shape
        num_classes=np.max(y)+1
        if self.W is None:#没有指定权重，则随机产生权重
            self.W=0.001*np.random.randn(dim,num_classes)
        loss_history=[]
        for it in range(num_iters):
            X_batch=None
            y_batch=None
            
            batch_idx=np.random.choice(num_train,batch_size,replace=True)
            
            X_batch=X[batch_idx]
            y_batch=y[batch_idx]
            
            loss,grad=self.loss(X_batch,y_batch,reg)
            
            loss_history.append(loss)
            
            self.W+=-1*learning_rate*grad
            
            if verbose and it%100==0:#显示中间迭代过程，输出是每次迭代的损失函数值
                print('iteration %d / %d: loss %f'%(it,num_iters,loss))
        if verbose:
            plt.plot(loss_history)#可视化损失函数 没有连续，离散，怎么连续？
            plt.show()
        return loss_history
    
    #预测类别
    def predict(self,X):
        y_pred=np.zeros(X.shape[1])
        scores=X.dot(self.W)
        y_pred=np.argmax(scores,axis=1)
        
        return y_pred
    
    def loss(self,X_batch,y_batch,reg):
        pass

#LinearClassifier 的子类 ，继承LinearClassifier类
class LinearSVM(LinearClassifier):
    def loss(self,X_batch,y_batch,reg):
        return svm_loss_vectorized(self.W,X_batch,y_batch,reg)#采用矩阵方式计算损失函数和梯度
            