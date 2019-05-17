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
    
    def feedback_rule(self, x,t):
        # Define the upright fixed point here.
        uf = np.array([10., 10.])
        xf = np.array([0., 0., 0., math.pi, 0., 0., 0., 0.])

        u = np.ones(2) * 15.
        K = 1.0

        ###### TODO ######
        # estimate appropriate u

        return u

    
class LQRController():
    
    def __init__(self, quadrotor_plant, 
                 xf = np.array([0., 0., 0., math.pi, math.pi, 0., 0., 0., 0., 0.])):
        self.plant = quadrotor_plant
        self.mb = self.plant.mb
        self.lb = self.plant.lb
        self.Ib = self.plant.Ib
        self.m1 = self.plant.m1
        self.l1 = self.plant.l1
        self.I1 = self.plant.I1
        self.m1 = self.plant.m2
        self.l1 = self.plant.l2
        self.I1 = self.plant.I2
        self.g = self.plant.g
        self.input_max = self.plant.input_max
        
        self.uf = np.array([15., 15.])
        self.xf = xf
        
        A, B = quadrotor_plant.GetLinearizedDynamics(self.uf, self.xf)
        
        self.Q = np.diag([1., 1., 2.1, 2., 2., 1., 1., 2., 2., 2.])
        self.R = np.diag([1., 1.])*.5
        
        from pydrake.all import LinearQuadraticRegulator
        self.K, self.S = LinearQuadraticRegulator(A, B, self.Q, self.R)

    def feedback_rule(self, x, t):
        u = self.uf
        
        x_d = x-self.xf

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
        self.m1 = self.plant.m2
        self.l1 = self.plant.l2
        self.I1 = self.plant.I2
        self.g = self.plant.g
        self.input_max = self.plant.input_max
        
        self.uf = uf
        self.xf = xf
        self.kp1 = 10
        self.kp2 = 10
        self.kb = 8.15

        self.uf = np.array([15., 15.])
        self.xf = xf
        
        A, B = quadrotor_plant.GetLinearizedDynamics(self.uf, self.xf)
        
        self.Q = np.diag([1., 1., 2.1, 2., 2., 1., 1., 2., 2., 2.])
        self.R = np.diag([1., 1.])*.5
        
        from pydrake.all import LinearQuadraticRegulator
        self.K, self.S = LinearQuadraticRegulator(A, B, self.Q, self.R)
        
        self.lqr_switch = 0
        self.J = 100000
        
    def feedback_rule(self, x, t):
        u = self.uf
        
        if self.lqr_switch == 1:
            if t>0.1:
                x_d = x-self.xf
                u = self.uf - np.dot(self.K,x_d)
                u[0] = max(-self.input_max, min(self.input_max, u[0]))
                u[1] = max(-self.input_max, min(self.input_max, u[1]))
            else:
                u[0] =10
                u[1] =10
            return u   
        
        dt = 0.005
        
        kp1 = self.kp1
        kp2 = self.kp2

        kb = self.kb
        Ib = self.Ib
        lb = self.lb
        g = self.g
        
        thetab = x[2]
        theta1 = x[3]
        theta2 = x[4]
        thetab_dot  =x[7]
        theta1_dot  =x[8]
        theta2_dot = x[9]
        
        theta1_f = self.xf[3]
        theta2_f = self.xf[3]
        
        F_norm = kp1*(theta1_f-theta1)+kp2*(theta2_f-theta2) -kb*theta1_dot
        
        if thetab > -np.pi:
            thetab += np.pi*2
            
        thetaA = 0.5*(theta1+theta2)
        thetaA_dot =  0.5*(theta1_dot+theta2_dot)
        calcA = (F_norm*np.cos(thetaA)+g)/np.cos(thetab)

        calcB = Ib *(thetab_dot-thetaA_dot)/(lb*dt)
        
        u = self.uf
        u[0] = (calcA+calcB)/2
        u[1] = (calcA-calcB)/2
        
        x_delta = x - self.xf
        J = x_delta.dot(self.S).dot(x_delta)

        if J < self.J:
            self.J = J
        
        if J < 82.2 and self.lqr_switch == 0:
            print("Switch lqr at J = {} at t = {}".format(J,t))
            print(x)
            self.lqr_switch =1

        u = np.clip(u, -self.input_max, self.input_max)

        return u
