# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:15:32 2018

@author: GFX
"""

#1、SVM线性分类器的构建

#1.1 损失函数和梯度计算
import numpy as np
from random import shuffle

#数值方式计算损失函数和梯度
#W:(D,K)维权重 ，K为类别数，D表示每个图像维度
#，(N，D)维输入图像，N表示图像数，每个图像都是D*1的列向量
#y:N维标签 
#reg:正则化系数
def svm_loss_naive(W,X,y,reg):
    dW=np.zeros(W.shape)
    num_classes=W.shape[1]
    num_train=X.shape[0]
    loss=0.0
    
    for i in range(num_train):
        scores=X[i].dot(W)
        correct_class_score=scores[y[i]] #计算正确分类数
        for j in range(num_classes):
            if j==y[i]:
                continue
            margin=scores[j]-correct_class_score+1 #delta=1 f(xi;W)j-f(xi;W)yi+delta
            if margin>0:    #max(0,f(xi;W)j-f(xi;W)yi+delta)
                loss+=margin
                dW[:,j]+=X[i].T
                dW[:,y[i]]+=-X[i].T
    loss /= num_train   #1/N*sum(max(....))
    dW /= num_train
    loss+=0.5*reg*np.sum(W*W)
    dW+=reg*W
    return loss,dW

#采用矩阵方式计算损失函数和梯度
def svm_loss_vectorized(W,X,y,reg):
    num_train=X.shape[0]
    num_classes=W.shape[1]
    scores=X.dot(W)
    correct_class_scores=scores[range(num_train),list(y)].reshape(-1,1)#-1说明要根据剩下的参数维度来计算当前维度
    margins=np.maximum(0,scores-correct_class_scores+1)
    margins[range(num_train),list(y)]=0
    loss=np.sum(margins)/num_train+0.5*reg*np.sum(W*W)
    
    coeff_mat=np.zeros((num_train,num_classes))
    coeff_mat[margins>0]=1
    coeff_mat[range(num_train),list(y)]=-np.sum(coeff_mat,axis=1)
    
    dW=(X.T).dot(coeff_mat)
    dW=dW/num_train+reg*W
    return loss,dW

