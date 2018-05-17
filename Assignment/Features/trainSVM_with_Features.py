# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:59:47 2018

@author: GFX
"""

# =============================================================================
# import numpy as np
# =============================================================================
from linear_classifer import LinearSVM
from DataPreprocess import *

learning_rates=[1e-9,1e-8,1e-7]
regularization_strengths=[(5+i)*1e6 for i in range(-3,4)]

results={}
best_val=-1
best_svm=None

for rs in regularization_strengths:
    for lr in learning_rates:
        svm=LinearSVM()
        loss_hist=svm.train(X_train_feats,y_train,lr,rs,num_iters=6000)
        y_train_pred=svm.predict(X_train_feats)
        train_accuracy=np.mean(y_train==y_train_pred)
        y_val_pred=svm.predict(X_val_feats)
        val_accuracy=np.mean(y_val==y_val_pred)
        if val_accuracy > best_val:
            best_val=val_accuracy
            best_svm=svm
        results[(lr,rs)]=train_accuracy,val_accuracy
        
for lr,reg in sorted(results):
    train_accuracy,val_accuracy=results[(lr,reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' %(lr,reg,train_accuracy,val_accuracy))
    
print('best validation accuracy achieved during cross‐validation: %f' %best_val)        