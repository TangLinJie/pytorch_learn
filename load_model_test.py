# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:25:22 2018

@author: 唐琳杰
"""
import torch 
def load_module():
    return torch.load('model.pkl')

model_load=load_module()
out=model_load(torch.randn(1,3,32,32))
print(out.data)