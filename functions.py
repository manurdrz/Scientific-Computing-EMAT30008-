# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:16:36 2023

@author: manue
"""
import numpy as np


def fo(x, t):
    dxdt = np.array([x])
    return dxdt

def true_fo_solution(t):
    x = np.exp(t)
    return x

def so(x,t):
    u,v = x
    dudt = v
    dvdt = -u
    dxdt = np.array([dudt, dvdt])
    return dxdt

def true_so_solution(t):
    u = np.cos(t)+ np.sin(t)
    v = np.cos(t)- np.sin(t)
    x = [u, v]
    return x

