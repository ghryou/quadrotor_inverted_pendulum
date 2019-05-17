# -*- coding: utf8 -*-
#!/usr/bin/env python
__author__ = "Gilhyun Ryou and Seong Ho Yeon"
__email__ = "ghryou@mit.edu, syeon@mit.edu"

import math
import numpy as np
import matplotlib.pyplot as plt

def analyze_energy(plant, state_log, input_log):
    mb = plant.mb
    lb = plant.lb # half of body length
    m1 = plant.m1
    l1 = plant.l1
    g = plant.g
    Ib = plant.Ib
    I1 = plant.I1
    
    def get_Kb(idx):
        xb_dot = state_log.data()[4][idx]
        yb_dot = state_log.data()[5][idx]
        thetab_dot = state_log.data()[6][idx]
        
        Kb = 0.5*mb*(xb_dot**2+yb_dot**2)+0.5*Ib*(thetab_dot**2)
        return Kb
    
    def get_K1(idx):
        theta1 = state_log.data()[3][idx]
        theta1_dot =  state_log.data()[7][idx]
        
        x1_dot = state_log.data()[4][idx] + l1*theta1_dot*np.cos(theta1)
        y1_dot = state_log.data()[5][idx] + l1*theta1_dot*np.sin(theta1)
        
        K1 = 0.5*m1*(x1_dot**2+y1_dot**2)+0.5*I1*(theta1_dot**2)    
        
        return K1
    
    def get_Pb(idx):
        yb = state_log.data()[1][idx]
        Pb = mb*g*yb
        return Pb
    
    def get_P1(idx):
        theta1 = state_log.data()[3][idx]
        y1 = state_log.data()[1][idx] - l1*np.cos(theta1)
        Pb = mb*g*y1
        return Pb

    def get_Pdelta(idx):
        theta1 = state_log.data()[3][idx]
        y_d = - l1*np.cos(theta1)
        Pdelta = mb*g*y_d
        return Pdelta
    
    n_step = state_log.sample_times().shape[0]
    
    E_size =6 
    E = np.zeros((n_step,E_size),dtype = float)
    E_name = ['Kb', 'K1', 'Pb', 'P1', 'Pdelta', 'E_total']
    for i in range(n_step):
        E[i,0] = get_Kb(i)
        E[i,1] = get_K1(i)
        E[i,2] = get_Pb(i)
        E[i,3] = get_P1(i)
        E[i,4] = get_Pdelta(i)
        E[i,5] = E[i,0]+E[i,1]+E[i,2]+E[i,3]
        
     
    # Visualize state and input traces
    t_stamp_i = 0
    t_stamp_f = -1
    
    fig = plt.figure().set_size_inches(12, 10)
    for i in range(E_size):
        plt.subplot(E_size, 1, i+1)
        plt.plot(state_log.sample_times()[t_stamp_i:t_stamp_f], E[t_stamp_i:t_stamp_f,i])
        plt.grid(True)
        plt.ylabel(E_name[i])
    