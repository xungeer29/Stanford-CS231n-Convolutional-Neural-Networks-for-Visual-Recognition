# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:53:28 2018

@author: GFX
"""

from DataPreprocess import *
from neural_net import TwoLayerNet

input_dim=X_train_feats.shape[1]
hidden_dim=500
num_classes=10

net=TwoLayerNet(input_dim,hidden_dim,num_classes)

results={}
best_val=-1
best_net=None

learning_rates = [1e-2, 1e-1, 5e-1, 1, 5]
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]

for lr in learning_rates:
    for reg in regularization_strengths:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        # Train the network
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,num_iters=1500, batch_size=200,
                          learning_rate=lr, learning_rate_decay=0.95,reg=reg, verbose=False)
        val_acc = (net.predict(X_val_feats) == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_net = net
        results[(lr, reg)] = val_acc
        
for lr, reg in sorted(results):
    val_acc = results[(lr, reg)]
    print('lr %e reg %e val accuracy: %f' % (lr, reg, val_acc))
    
print('best validation accuracy achieved during cross‐validation: %f' %best_val)