# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:23:45 2018

@author: GFX
"""

import random
#梯度检验
#将数值梯度法与分析梯度法计算结果进行比较
def grad_check_sparse(f,x,analytic_grad,num_checks=10,h=1e-5):
    for i in range(num_checks):
        ix=tuple([random.randrange(m) for m in x.shape])
        
        oldval=x[ix]
        x[ix]=oldval+h
        fxph=f(x)
        x[ix]=oldval-h
        fxmh=f(x)
        x[ix]=oldval
        
        grad_numerical=(fxph-fxmh)/(2*h)
        grad_analytic=analytic_grad[ix]
        rel_error=abs(grad_numerical-grad_analytic)/(abs(grad_numerical)+abs(grad_analytic))
        print('numerical: %f analytic: %f,relative error:%e'%(grad_numerical,grad_analytic,rel_error))