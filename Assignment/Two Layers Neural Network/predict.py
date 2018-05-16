# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:14:34 2018

@author: GFX
"""
from best_net import best_net
from train import *

test_acc=(best_net.predict(X_test)==y_test).mean()
print('test accuracy:' , test_acc)#0.502