# -*- coding: utf8 -*-
#!/usr/bin/env python
__author__ = "Gilhyun Ryou and Seong Ho Yeon"
__email__ = "ghryou@mit.edu, syeon@mit.edu"

import numpy as np
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pydrake.all import (
        Context,
        DiagramBuilder,
        LeafSystem,
        PortDataType,
    )
import scipy.interpolate

from underactuated import PyPlotVisualizer
from quadrotor_pendulum import *

# Custom visualizer for a quadrotor pendulum

def populate_square_vertices(edge_length):
    return np.array([[-edge_length, -edge_length, edge_length, edge_length, -edge_length],
                     [-edge_length, edge_length, edge_length, -edge_length, -edge_length]])

def populate_rectangle_vertices(width, height):
    return np.array([[-width, -width, width, width, -width],
                     [-height, height, height, -height, -height]])

def populate_disk_vertices(radius, width, N):
    av = np.linspace(0, 2*math.pi, N)

    outer_circle_x = np.array([radius*math.cos(v) for v in av])
    outer_circle_y = np.array([radius*math.sin(v) for v in av])
    inner_circle_x = np.array([(radius-width)*math.cos(v) for v in av])
    inner_circle_y = np.array([(radius-width)*math.sin(v) for v in av])

    all_x = np.hstack([outer_circle_x, inner_circle_x[::-1], outer_circle_x[0]])
    all_y = np.hstack([outer_circle_y, inner_circle_y[::-1], outer_circle_y[0]])
    return np.vstack([all_x, all_y])

def populate_circle_vertices(radius, N):
    av = np.linspace(0, 2*math.pi, N)

    circle_x = np.array([radius*math.cos(v) for v in av])
    circle_y = np.array([radius*math.sin(v) for v in av])
    
    return np.vstack([circle_x, circle_y])

def rotmat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

class QuadrotorPendulumVisualizer(PyPlotVisualizer):

    def __init__(self, quadrotor_pendulum):
        PyPlotVisualizer.__init__(self, figsize=(8,8))
        self.screen_size = 15
        self.set_name('Quadrotor Pendulum Visualizer')
        self._DeclareInputPort(PortDataType.kVectorValued, 2) # full output of its controller
        self._DeclareInputPort(PortDataType.kVectorValued, 10) # full output of the pendulum visualizer
        
        lim = (quadrotor_pendulum.lb+quadrotor_pendulum.l1)*self.screen_size
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        
        self.drone_height = 0.04
        self.lb = quadrotor_pendulum.lb
        self.quadrotor_pts = populate_rectangle_vertices(self.lb, self.drone_height)
        self.quadrotor = self.ax.fill(self.quadrotor_pts[0, :], self.quadrotor_pts[1, :], zorder=1, facecolor=(.3, .6, .4), edgecolor='k', closed=True)
        
        self.propeller1_pts = populate_square_vertices(self.drone_height*1.2)
        self.propeller1_pts[0,:] += self.lb
        self.propeller1 = self.ax.fill(self.propeller1_pts[0, :], self.propeller1_pts[1, :], zorder=1, facecolor=(.6, .6, .4), edgecolor='k', closed=True)
        
        self.propeller2_pts = populate_square_vertices(self.drone_height*1.2)
        self.propeller2_pts[0,:] -= self.lb
        self.propeller2 = self.ax.fill(self.propeller2_pts[0, :], self.propeller2_pts[1, :], zorder=1, facecolor=(.6, .6, .4), edgecolor='k', closed=True)
    
    
        self.pendulum_radius = 0.01
        self.l1 = quadrotor_pendulum.l1
        self.pendulum1_pts = populate_rectangle_vertices(self.pendulum_radius, self.l1)
        self.pendulum1_pts[1,:] -= self.l1
        self.pendulum1 = self.ax.fill(self.pendulum1_pts[0, :], self.pendulum1_pts[1, :], zorder=1, facecolor=(.0, .0, .0), edgecolor='k', closed=True)
        
        self.pendulum1_com_pts = populate_circle_vertices(self.pendulum_radius*4, 10)
        self.pendulum1_com_pts[1,:] -= self.l1
        self.pendulum1_com = self.ax.fill(self.pendulum1_com_pts[0, :], self.pendulum1_com_pts[1, :], zorder=3, facecolor='tab:red', edgecolor='k', closed=True)

        
        self.n_arrow = 2
        self.arrow_size = 0.6
        self.Q = self.ax.quiver(np.zeros(self.n_arrow),
                   np.zeros(self.n_arrow),
                   np.zeros(self.n_arrow),
                   np.zeros(self.n_arrow),
                   pivot='tail',
                   color='r',
                   units='xy',
                   scale=1.0)
            
        # todo: input visualization?

    def draw(self, context):
        if isinstance(context, Context):
            xb = self.EvalVectorInput(context, 1).get_value()[0]
            yb = self.EvalVectorInput(context, 1).get_value()[1]
            thetab = self.EvalVectorInput(context, 1).get_value()[2]
            theta1 = self.EvalVectorInput(context, 1).get_value()[3]
            
            u1 = self.EvalVectorInput(context, 0).get_value()[0]
            u2 = self.EvalVectorInput(context, 0).get_value()[1]
            
        else:
            xb = context[1][0]
            yb = context[1][1]
            thetab = context[1][2]
            theta1 = context[1][3]
            
            u1 = context[0][0]
            u2 = context[0][1]
            
        rotmat_thetab = rotmat(thetab)
        rotmat_theta1 = rotmat(theta1)

        path = self.quadrotor[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_thetab, self.quadrotor_pts).T
        xc = np.mean(path.vertices[:-1,0])
        yc = np.mean(path.vertices[:-1,1])
        path.vertices[:,0] += (xb-xc)
        path.vertices[:,1] += (yb-yc)
        
        
        path = self.propeller1[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_thetab, self.propeller1_pts).T
        path.vertices[:,0] += (xb-xc)
        path.vertices[:,1] += (yb-yc)
        
        path = self.propeller2[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_thetab, self.propeller2_pts).T
        path.vertices[:,0] += (xb-xc)
        path.vertices[:,1] += (yb-yc)
        
        path = self.pendulum1[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_theta1, self.pendulum1_pts).T
        path.vertices[:,0] += (xb-xc)
        path.vertices[:,1] += (yb-yc)
        
        path = self.pendulum1_com[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_theta1, self.pendulum1_com_pts).T
        path.vertices[:,0] += (xb-xc)
        path.vertices[:,1] += (yb-yc)

        new_X = np.ones(self.n_arrow)*100 # guaranteed off-screen
        new_Y = np.ones(self.n_arrow)*100
        new_U = np.zeros(self.n_arrow)
        new_V = np.zeros(self.n_arrow)
#         for i in range(self.n_arrow):
        new_X[0] = (xb-xc) - math.cos(thetab)*self.lb
        new_Y[0] = (yb-yc) - math.sin(thetab)*self.lb
        new_X[1] = (xb-xc) + math.cos(thetab)*self.lb
        new_Y[1] = (yb-yc) + math.sin(thetab)*self.lb
#         for i in range(self.n_arrow):
        new_U[0] = -math.sin(thetab)*u1/10*self.arrow_size
        new_V[0] = math.cos(thetab)*u1/10*self.arrow_size
        new_U[1] = -math.sin(thetab)*u2/10*self.arrow_size
        new_V[1] = math.cos(thetab)*u2/10*self.arrow_size
        
        self.Q.set_offsets(np.vstack([new_X, new_Y]).T)
        self.Q.set_UVC(new_U, new_V)
        
        
    def animate(self, input_log, state_log, rate, resample=True, repeat=False):
        # log - a reference to a pydrake.systems.primitives.SignalLogger that
        # constains the plant state after running a simulation.
        # rate - the frequency of frames in the resulting animation
        # resample -- should we do a resampling operation to make
        # the samples more consistent in time? This can be disabled
        # if you know the sampling rate is exactly the rate you supply
        # as an argument.
        # repeat - should the resulting animation repeat?
        t = state_log.sample_times()
        u = input_log.data()
        x = state_log.data()

        if resample:
            t_resample = np.arange(0, t[-1], 1./rate)
            u = scipy.interpolate.interp1d(input_log.sample_times(), u, kind='linear', axis=1)(t_resample)
            x = scipy.interpolate.interp1d(t, x, kind='linear', axis=1)(t_resample)
            t = t_resample

        animate_update = lambda i: self.draw([u[:, i], x[:, i]])
        ani = animation.FuncAnimation(self.fig, animate_update, t.shape[0], interval=1000./rate, repeat=repeat)
        return ani