# -*- coding: utf8 -*-
#!/usr/bin/env python
__author__ = "Gilhyun Ryou and Seong Ho Yeon"
__email__ = "ghryou@mit.edu, syeon@mit.edu"

import math
import numpy as np

from pydrake import math as dm
from pydrake.all import MathematicalProgram, Solve, IpoptSolver, SolverOptions

class SampleController():
    
    def __init__(self, quadrotor_plant):
        self.plant = quadrotor_plant
        self.mb = self.plant.mb
        self.lb = self.plant.lb
        self.Ib = self.plant.Ib
        self.m1 = self.plant.m1
        self.l1 = self.plant.l1
        self.I1 = self.plant.I1
        self.g = self.plant.g
        self.input_max = self.plant.input_max
    
    def feedback_rule(self, x, t):
        # Define the upright fixed point here.
        uf = np.array([10., 10.])
        xf = np.array([0., 0., 0., math.pi, 0., 0., 0., 0.])

        u = np.ones(2) * 10.
        K = 1.0

        ###### TODO ######
        # estimate appropriate u

        return u

    
class LQRController():
    
    def __init__(self, quadrotor_plant, 
                 uf = np.array([10., 10.]), 
                 xf = np.array([0., 0., 0., math.pi, 0., 0., 0., 0.]) ):
        self.plant = quadrotor_plant
        self.mb = self.plant.mb
        self.lb = self.plant.lb
        self.Ib = self.plant.Ib
        self.m1 = self.plant.m1
        self.l1 = self.plant.l1
        self.I1 = self.plant.I1
        self.g = self.plant.g
        self.input_max = self.plant.input_max
        
        self.uf = uf
        self.xf = xf
        
        A, B = quadrotor_plant.GetLinearizedDynamics(self.uf, self.xf)
    
        Q = np.eye(8)
        R = np.eye(2)*2
        
        from pydrake.all import LinearQuadraticRegulator
        self.K, self.S = LinearQuadraticRegulator(A, B, Q, R)
        
        self.A = A
        self.B = B
        
    def feedback_rule(self, x, t):
        u = self.uf
        
        x_d = x-self.xf
#         x_d[2] = (x_d[2]+np.pi-1e-6)%(2.*np.pi)-np.pi+1e-6
#         x_d[3] = (x_d[3]+np.pi-1e-6)%(2.*np.pi)-np.pi+1e-6

        u = self.uf - np.dot(self.K,x_d)
        u[0] = max(-self.input_max, min(self.input_max, u[0]))
        u[1] = max(-self.input_max, min(self.input_max, u[1]))
            
        return u
    

class OurController():
    
    def __init__(self, quadrotor_plant, 
                 uf = np.array([10., 10.]), 
                 xf = np.array([0., 0., 0., math.pi, 0., 0., 0., 0.]) ):
        self.plant = quadrotor_plant
        self.mb = self.plant.mb
        self.lb = self.plant.lb
        self.Ib = self.plant.Ib
        self.m1 = self.plant.m1
        self.l1 = self.plant.l1
        self.I1 = self.plant.I1
        self.g = self.plant.g
        self.input_max = self.plant.input_max
        
        self.uf = uf
        self.xf = xf
        self.kp = 10
        self.kb = 8
        
        A, B = quadrotor_plant.GetLinearizedDynamics(self.uf, self.xf)
    
        Q = np.eye(8)
        R = np.eye(2)*2
        
        from pydrake.all import LinearQuadraticRegulator
        self.K, self.S = LinearQuadraticRegulator(A, B, Q, R)
        
        self.A = A
        self.B = B
        self.lqr_switch = 0
        
        self.cnt =0
        self.u_step = 12
        self.J = 1000000
        self.switch_t = 0
        
    def feedback_rule(self, x,t):
        u = self.uf
        if self.cnt==0:
            u[0] = 0
            u[1] = 0
            self.cnt+=1
            return u
        elif self.cnt < 4:
            u[0] = -self.u_step
            u[1] = +self.u_step
            self.cnt+=1
            return u
        elif self.cnt == 4:
            u[0] = 0
            u[1] = 0
            self.cnt+=1
            return u
        elif self.cnt < 8:
            u[0] = +self.u_step
            u[1] = -self.u_step
            self.cnt+=1
            return u
        elif self.cnt == 8:
            u[0] = 0
            u[1] = 0
            self.cnt+=1
            return u   
        
        if self.lqr_switch == 1:
            if t>self.switch_t+0.015:
                x_d = x-self.xf
                u = self.uf - np.dot(self.K,x_d)
                u[0] = max(-self.input_max, min(self.input_max, u[0]))
                u[1] = max(-self.input_max, min(self.input_max, u[1]))
            else:
                u[0] = 10
                u[1] = 10
            return u   
        
        dt = 0.005
        kp = self.kp
        kb = self.kb
        Ib = self.Ib
        lb = self.lb
        g = self.g
        thetab = x[2]
        theta1 = x[3]
        thetab_dot = x[6]
        theta1_dot = x[7]
        theta1_f = self.xf[3]
        F_norm = kp*(theta1_f-theta1)-kb*theta1_dot
        
        calcA = (F_norm*np.cos(theta1+theta1_dot*dt/2)+g)/np.cos(thetab+thetab_dot*dt/2)
        calcB = Ib *(thetab_dot-theta1_dot)/(lb*dt)
        
        u = self.uf
        u[0] = (calcA+calcB)/2
        u[1] = (calcA-calcB)/2
        
        x_delta = x - self.xf
        J = x_delta.dot(self.S).dot(x_delta)
        
        if self.J >J:
            self.J = J
                     
        elif self.lqr_switch == 0:
            self.switch_t  = t
            print("Switch lqr at J = {} at t = {}".format(J,t))
            print(x)
            self.lqr_switch =1

        u = np.clip(u, -self.input_max, self.input_max)

        return u
