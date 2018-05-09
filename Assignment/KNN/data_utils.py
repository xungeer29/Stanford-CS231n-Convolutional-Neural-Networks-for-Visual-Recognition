# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:41:23 2018

@author: GFX
"""

#数据载入

#采用的是cifar10数据集
#该数据集被分成5份训练集和1份测试集，
#每份有1000张32*32的RGB图，共有10类

import pickle
import numpy as np
import os

#载入训练集和测试集
def load_cifar_batch(filename):
    with open(filename,'rb') as f:
        datadict=pickle.load(f,encoding='bytes')
        x=datadict[b'data']
        y=datadict[b'labels']
        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y=np.array(y)
        return x,y

    
def load_cifar10(root):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(root,'data_batch_%d' %(b,))
        x,y=load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain=np.concatenate(xs)#将5份训练集转换成数组
    Ytrain=np.concatenate(ys)#将1份测试集转换成数组
    del x,y
    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch'))
    return Xtrain,Ytrain,Xtest,Ytest

#import  pickle
#import numpy as np
#import os
#def load_cifar_batch(filename):
#    with open(filename,'rb') as f :
#        datadict=pickle.load(f,encoding='bytes')
#        x=datadict[b'data']
#        y=datadict[b'labels']
#        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
#        y=np.array(y)
#        return x,y
#def load_cifar10(root):
#    xs=[]
#    ys=[]
#    for b in range(1,6):
#        f=os.path.join(root,'data_batch_%d' % (b,))
#        x,y=load_cifar_batch(f)
#        xs.append(x)
#        ys.append(y)
#    Xtrain=np.concatenate(xs) #1
#    Ytrain=np.concatenate(ys)
#    del x ,y
#    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch')) #2
#    return Xtrain,Ytrain,Xtest,Ytest