# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:20:42 2018

@author: GFX
"""

import numpy as np
from trainSVM_with_Features import best_svm
from DataPreprocess import X_test_feats,y_test

y_test_pred=best_svm.predict(X_test_feats)
test_accuracy=np.mean(y_test==y_test_pred)
print('test accuracy: %f' %test_accuracy)


import matplotlib.pyplot as plt
from DataPreprocess import X_test
#分类结果可视化
examples_per_calss=8
classes=['palne','car','bird','cat','deer','dog','frog','horse','ship','truck']

for cls,cls_name in enumerate(classes):
    idxs=np.where((y_test !=cls) & (y_test_pred==cls))[0]
    idxs=np.random.choice(idxs,examples_per_calss,replace=False)
    for i ,idx in enumerate(idxs):
        plt.subplot(examples_per_calss,len(classes),i*len(classes)+cls+1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i ==0:
            plt.title(cls_name)
plt.show()