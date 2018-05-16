# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:54:06 2018

@author: GFX
"""

#3.2 超参数调整
#将调整后的最优参数模型保存为best_net，方便下面调用，进行预测

#可视化结果分析：
#loss曲线大致呈线性下降，这表明我们设置的学习率可能太低；
#训练集和验证集的准确率很接近，表明我们的模型复杂度不够，也就是欠拟合；
#当然增加模型复杂度的话可能会导致过拟合

#通过交叉验证的方法来不断调整超参数
import numpy as np
from train import *#X_train,y_train,X_val,y_val ......
# =============================================================================
# from neural_net import TwoLayerNet#在train中已经import了
# =============================================================================
input_size=32*32*3
num_classes=10
hidden_size=[75,100,125]#输入层分别尝试75，100，125层大小
results={}
best_val_acc=0
best_net=None
learning_rates=np.array([0.7,0.8,0.9,1.0,1.1])*1e-3#学习率选项
regularization_strengths=[0.75,1.0,1.25]#正则化强度选项
print('running')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            net=TwoLayerNet(input_size,hs,num_classes)
            
            stats=net.train(X_train,y_train,X_val,y_val,num_iters=1500,batch_size=200,
                            learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=False)
            val_acc=(net.predict(X_val)==y_val).mean()#判断预测结果
# =============================================================================
#             val_acc=(net.predict(X_val)==y_val).mean()
# =============================================================================
            if val_acc >best_val_acc:#选择最好的参数
                bestval_acc=val_acc
                best_net=net
            results[(hs,lr,reg)]=val_acc
            
            print('finshed')

            for hs,lr,reg in sorted(results):
                val_acc=results[(hs,lr,reg)]
                print('hs %d lr %e reg %e val accuracy: %f' % (hs,lr,reg,val_acc))
    
print('best validation accuracy achieved during cross_validation: %f' %best_val_acc)   

#可视化学到的最优参数
from vis_utils import show_net_weights
    
show_net_weights(best_net)












         