# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:30:35 2018

@author: GFX
"""

# 3.1.1 loss和accuracy可视化


import matplotlib.pyplot as plt
from train import stats

plt.subplot(211)
plt.plot(stats['loss_history'])
plt.title('loss history')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(stats['train_acc_history'],label='train')
plt.plot(stats['val_acc_history'],label='val')
plt.title('classification accuracy history')
plt.xlabel('epoch')
plt.ylabel('classification accuracy')
plt.show()


# 3.1.2 权重可视化
#CS231n提供，可直接调用
#要求输入的 Xs 是四维的

# =============================================================================
# from cmath import sqrt#sqrt
# =============================================================================
from math import ceil, sqrt#ceil
import numpy as np
def visualize_grid(Xs, ubound=255.0, padding=1):#将图像向量重新转化为图像 
                                                #ubound：灰度级数上界 padding：图像之间的间隔
    (N, H, W, C) = Xs.shape#N:图像数目 H:高 W:宽 C:通道数
    grid_size = int(ceil(sqrt(N)))#ceil()向上取整 例如100个种类，变成10*10显示
    grid_height = H * grid_size + padding * (grid_size-1)#每个权重图像高度H*一列的图像数+两张图像之间的间隔*9
    grid_width = W * grid_size + padding * (grid_size-1)
    grid = np.zeros((grid_height, grid_width, C))#新建存储可视化权重的矩阵
    next_idx = 0
    y0, y1 = 0, H#y0,y1:每一个可视化权重图像高度的起始点与终止点
    for y in range(grid_size):
        x0, x1 = 0, W#每一个可视化权重图像宽上的起始点与终止点
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high-low)#通过比例计算灰度
                next_idx += 1
            x0 += W + padding#加上图像之间的间隔距离
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


#调用上述函数，可视化权重
from train import net
def show_net_weights(net):
    W1=net.params['W1']
    W1=W1.reshape(32,32,3,-1).transpose(3,0,1,2)################没看懂#########################
    plt.imshow(visualize_grid(W1,padding=3).astype('uint8'))
    plt.axis('off')
    plt.show()
    
show_net_weights(net)