#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:38:58 2020

@author: davi
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multilateration_algorithms import *
from shot import *

def cil2car(r, theta, z):
    ''' 
    changes from cylindrical coordinates to Cartesian coordinates
    obs: theta in radians
    '''
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x, y, z])

def car2sph(x, y, z):
    ''' 
    changes from Cartesian coordinates to spherical coordinates
    obs: theta, phi in degrees
    '''
    if ((x, y, z) == (0, 0, 0)): return np.zeros(3)
    epsilon = 1e-17 # avoid division by zero
    
    R = np.sqrt(x**2 + y**2 + z**2)
    theta_rad = np.arctan(y/(x + epsilon))
    phi_rad = np.arccos(z/(R + epsilon))
    
    theta = (theta_rad/(2*np.pi)) * 360 
    phi = (phi_rad/(2*np.pi)) * 360
    return np.array([R, theta, phi])

def sph2car(R, phi, theta):
    ''' 
    changes from spherical coordinates to Cartesian coordinates
    R : radius
    phi : azimuth angle in radians
    theta : elevation angle in radians
    
    return ndarray with x,y,z Cartesian coordinates
    '''
    x = R*np.sin(theta)*np.cos(phi)
    y = R*np.sin(theta)*np.sin(phi)
    z = R*np.cos(theta)
    return np.array([x, y, z])

# generate cilindrical sources cloud
r = np.linspace(1, 15, 15) # r is in [1,15] with steps of 1
theta = np.linspace(0, 2*np.pi, 24, endpoint=False) # r is in [0,360) with steps of 15 degrees
z = np.linspace(0, 6, 7) # z is in [-3,3] with steps of 1

cilinder = itertools.product(r, theta, z)

# generate spherical sources cloud
R = np.linspace(1,15,15)
phi = np.linspace(0, 2*np.pi,24, endpoint=False)
theta = np.linspace(0, np.pi/2, 12, endpoint=True)

nPoints = len(R) * len(phi) * len(theta)

# generate parallelepiped
X = np.linspace(-10,10,21)
Y = np.linspace(-10,10,21)
Z = np.linspace(0,10,11)

#nPoints = len(X) * len(Y) * len(Z)

#square (4 microphones)
square_array = np.zeros((4,3))

square_array[0] = np.array([-1, 1, 1])
square_array[1] = np.array([-1, -1, -1])
square_array[2] = np.array([1, 1, -1])
square_array[3] = np.array([1, -1, -1])
#square_array[4] = np.array([0, 0, -1])
#square_array[5] = np.array([-1, 1, -1])

#our array
our_array = np.zeros((4,3))

our_array[0] = np.array([-0.83, -0.48, -0.18])
our_array[1] = np.array([ 0.97, -0.03,  0.23])
our_array[2] = np.array([ 0.20,  0.20, -0.14])
our_array[3] = np.array([-0.03,  0.99, -0.12])

#our array2

our_array2 = np.array([[-0.12620505, -0.63363287, -0.15411901],
       [ 0.38171208, -0.03580296, -0.19768694],
       [ 0.02742979, -0.31217145, -0.08820106],
       [-0.48513401, -0.16910477, -0.27557907]])

#spherical (6 microphones)
spherical_array = np.zeros((6,3))
spherical_array[0] = np.array([0, 0, 1.5])
spherical_array[1] = np.array([1, 1, 1])
spherical_array[2] = np.array([1, -1, 1])
spherical_array[3] = np.array([-1, 1, 1])
spherical_array[4] = np.array([-1, -1, 1])
spherical_array[5] = np.array([0, 0, 0.5])

m = np.array([1, 1, 1, 1, 1, 1])
e = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# arrays evaluation
sampleRate = 120000
samples = int(sampleRate/2) 
arrays = (our_array, our_array2)
big_error = [[]]*len(arrays)

for i,array in enumerate(arrays):
    count=0
    aux = []
    l,_ = np.shape(array)
    #parallelepiped = itertools.product(X, Y, Z)
    semi_sphere = itertools.product(R, phi, theta)
    for (a, b, c) in semi_sphere:
        #source = np.array([a, b, c]) + 0.05*np.random.randn(3)
        source = sph2car(a, b, c) + 0.05*np.random.randn(3)
        
        toa = dist(source, array)/soundSpeed
        tdoa = toa[0] - toa[1:,] 
        
        tdoa = np.round(tdoa*sampleRate)/sampleRate
        '''rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
    
        def shot (t):
            decayConst = -10
            relu = (t>=0)
            index = np.sum(~relu)
            r = np.append(np.zeros(index), rand[0:rand.shape[0]-index])
            return r * np.exp(decayConst * t) * relu
        
        tdoa, _ = delayEstimator (shot, sampleRate, source, array, typeComb = "")'''
        result = MLE_HLS(array, tdoa, sampleRate, typeComb="")
        
        error = float(dist(source, result))
        #if dist(source, result) > 4.0: big_error.append(source)
        aux.append(error)
        
        count+=1
        if count%500==0: print(count)
    big_error[i] = aux
big_error = [sorted(elem) for elem in big_error]
big_error = np.array(big_error)
mle = np.sum(big_error**2, axis=1)/nPoints

plt.title("Distance distribution for arrays")
plt.plot(big_error[0], c='blue')
plt.plot(big_error[1], c='green')
plt.legend([f"square array(MLE={mle[0]:.2f})", f"our array(MLE={mle[1]:.2f})"])
plt.xlabel("source number")
plt.ylabel("distance")
plt.yscale("log")
plt.grid()
plt.show()

