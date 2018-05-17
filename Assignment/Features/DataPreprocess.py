# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:37:08 2018

@author: GFX
"""

import numpy as np
from data_utils import load_cifar10
from featuresExtract import hog_feature,color_histogram_hsv,extract_features


def get_cifar_data(num_training=49000,num_validation=1000,num_test=1000):
    cifar10_dir='datasets'
    X_train,y_train,X_test,y_test=load_cifar10(cifar10_dir)
    # 验证集
    mask=range(num_training,num_training+num_validation)
    X_val=X_train[mask]
    y_val=y_train[mask]
    #训练集
    mask=range(num_training)
    X_train=X_train[mask]
    y_train=y_train[mask]
    #测试集
    mask=range(num_test)
    X_test=X_test[mask]
    y_test=y_test[mask]
    
    return X_train,y_train,X_val,y_val,X_test,y_test

X_train,y_train,X_val,y_val,X_test,y_test=get_cifar_data()
num_color_bins = 10
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img,nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

mean_feat=np.mean(X_train_feats,axis=0,keepdims=True)
X_train_feats-=mean_feat
X_val_feats-=mean_feat
X_test_feats-=mean_feat
std_feat=np.std(X_train_feats,axis=0,keepdims=True)
X_train_feats/=std_feat
X_val_feats/=std_feat
X_test_feats/=std_feat

X_train_feats=np.hstack([X_train_feats,np.ones((X_train_feats.shape[0],1))])
X_val_feats=np.hstack([X_val_feats,np.ones((X_val_feats.shape[0],1))])
X_test_feats=np.hstack([X_test_feats,np.ones((X_test_feats.shape[0],1))])