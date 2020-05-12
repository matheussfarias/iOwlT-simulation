#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:23:56 2020

@author: davi
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multilateration_algorithms import *
import time

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

sampleRate = 32000
#generate function that simulates the shot emmited
samples = int(sampleRate/2)

cilinder = itertools.product(r, theta, z)

# generate spherical sources cloud
R = np.linspace(1,15,15)
phi = np.linspace(0, 2*np.pi,24, endpoint=False)
theta = np.linspace(0, np.pi/2, 12, endpoint=True)

semi_sphere = itertools.product(R, phi, theta)

# generate parallelepiped
x = np.linspace(-10,10,21)
y = np.linspace(-10,10,21)
z = np.linspace(0,10,11)

parallelepiped = itertools.product(x, y, z)

#square (4 microphones)
square_array = np.zeros((6,3))

square_array[0] = np.array([0, 0, 1])
square_array[1] = np.array([-1, -1, -1])
square_array[2] = np.array([1, 1, -1])
square_array[3] = np.array([1, -1, -1])
square_array[4] = np.array([0, 0, -1])
square_array[5] = np.array([-1, 1, -1])

#piramidal (4 microphones)
#piramidal (6 microphones)
#spherical (6 microphones)
spherical_array = np.zeros((6,3))
spherical_array[0] = np.array([0, 0, 1.5])
spherical_array[1] = np.array([1, 1, 1])
spherical_array[2] = np.array([1, -1, 1])
spherical_array[3] = np.array([-1, 1, 1])
spherical_array[4] = np.array([-1, -1, 1])
spherical_array[5] = np.array([0, 0, 0.5])

array = square_array
array_ax = plt.axes(projection='3d')
array_ax.set_xlim([-1.5, 1.5])
array_ax.set_xlim([-1.5, 1.5])
array_ax.set_xlim([-1.5, 1.5])
array_ax.scatter3D(array[:,0], array[:,1], array[:,2], c='darkblue', depthshade=False , s=40)
for i in range(array.shape[0]):
    array_ax.text(array[i,0], array[i,1], array[i,2]+0.1, f"mic {str(i)}", zorder=1)
plt.title("Disposição espacial dos microfones", fontsize=12)
plt.show()

m = np.array([1, 1, 1, 1, 1, 1])
e = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# arrays evaluation
count=0

r_5cm = []
r_20cm = []
r_1m = []
r_2m = []
r_error = []

phi_1 = []
phi_5 = []
phi_10 = []
phi_20 = []
phi_error = []

theta_1 = []
theta_5 = []
theta_10 = []
theta_20 = []
theta_error = []


for (R, phi, theta) in semi_sphere:
    source = sph2car(R, phi, theta) + np.random.randn(3)*0.1
    source = np.round(source, decimals=2)
    #source = np.array([R, phi, theta])

    rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1    
    delays, _ = delayEstimator (rand, sampleRate, source, square_array, m, e, typeComb = "")
    result = MLE_HLS(square_array, delays, sampleRate)
    sph_source = car2sph(source[0], source[1], source[2])
    sph_result = car2sph(result[0], result[1], result[2])
    
    error = np.abs(sph_source - sph_result)
    
    if error[0]<0.05: r_5cm.append(source)
    elif error[0]<0.2: r_20cm.append(source)
    elif error[0]<1.0: r_1m.append(source)
    elif error[0]<2.0: r_2m.append(source)
    else: r_error.append(source)

    if error[1]<1: phi_1.append(source)
    elif error[1]<5: phi_5.append(source)
    elif error[1]<10: phi_10.append(source)
    elif error[1]<20: phi_20.append(source)
    else: phi_error.append(source)
    
    if error[2]<1: theta_1.append(source)
    elif error[2]<5: theta_5.append(source)
    elif error[2]<10: theta_10.append(source)
    elif error[2]<20: theta_20.append(source)
    else: theta_error.append(source)

    count+=1
    if count%500==0: print(count)
r_5cm = np.stack(r_5cm, 0) if r_5cm else []
r_20cm = np.stack(r_20cm, 0) if r_20cm else []
r_1m = np.stack(r_1m, 0) if r_1m else []
r_2m = np.stack(r_2m, 0) if r_2m else []
r_error = np.stack(r_error, 0) if r_error else[]

phi_1 = np.stack(phi_1, 0) if phi_1 else []
phi_5 = np.stack(phi_5, 0) if phi_5 else []
phi_10 = np.stack(phi_10, 0) if phi_10 else []
phi_20 = np.stack(phi_20, 0) if phi_20 else []
phi_error = np.stack(phi_error, 0) if phi_error else [] 

theta_1 = np.stack(theta_1, 0) if theta_1 else []
theta_5 = np.stack(theta_5, 0) if theta_5 else []
theta_10 = np.stack(theta_10, 0) if theta_10 else []
theta_20 = np.stack(theta_20, 0) if theta_20 else []
theta_error = np.stack(theta_error, 0) if theta_error else []

fontsize = 12

r_ax = plt.axes(projection='3d')
if type(r_5cm) == np.ndarray : r_ax.scatter3D(r_5cm[:,0], r_5cm[:,1], r_5cm[:,2], c='lime', s=4)
if type(r_20cm) == np.ndarray: r_ax.scatter3D(r_20cm[:,0], r_20cm[:,1], r_20cm[:,2], c='greenyellow', s=4)
if type(r_1m) == np.ndarray: r_ax.scatter3D(r_1m[:,0], r_1m[:,1], r_1m[:,2], c='yellow', s=4)
if type(r_2m) == np.ndarray: r_ax.scatter3D(r_2m[:,0], r_2m[:,1], r_2m[:,2], c='darkorange', s=4)
if type(r_error) == np.ndarray: r_ax.scatter3D(r_error[:,0], r_error[:,1], r_error[:,2], c='red', s=4)
plt.title('Espaço de taxas de acerto de R(raio)', fontsize = fontsize)

plt.show()

phi_ax = plt.axes(projection='3d')
if type(phi_1) == np.ndarray: phi_ax.scatter3D(phi_1[:,0], phi_1[:,1], phi_1[:,2], c='lime', s=4)
if type(phi_5) == np.ndarray: phi_ax.scatter3D(phi_5[:,0], phi_5[:,1], phi_5[:,2], c='greenyellow', s=4)
if type(phi_10) == np.ndarray: phi_ax.scatter3D(phi_10[:,0], phi_10[:,1], phi_10[:,2], c='yellow', s=4)
if type(phi_20) == np.ndarray: phi_ax.scatter3D(phi_20[:,0], phi_20[:,1], phi_20[:,2], c='darkorange', s=4)
if type(phi_error) == np.ndarray: phi_ax.scatter3D(phi_error[:,0], phi_error[:,1], phi_error[:,2], c='red', s=4)
plt.title(f'Espaço de taxas de acerto de {chr(934)}(ângulo de azimute)', fontsize = fontsize)
plt.show()

theta_ax = plt.axes(projection='3d')
if type(theta_1) == np.ndarray: theta_ax.scatter3D(theta_1[:,0], theta_1[:,1], theta_1[:,2], c='lime', s=4)
if type(theta_5) == np.ndarray: theta_ax.scatter3D(theta_5[:,0], theta_5[:,1], theta_5[:,2], c='greenyellow', s=4)
if type(theta_10) == np.ndarray: theta_ax.scatter3D(theta_10[:,0], theta_10[:,1], theta_10[:,2], c='yellow', s=4)
if type(theta_20) == np.ndarray: theta_ax.scatter3D(theta_20[:,0], theta_20[:,1], theta_20[:,2], c='darkorange', s=4)
if type(theta_error) == np.ndarray: theta_ax.scatter3D(theta_error[:,0], theta_error[:,1], theta_error[:,2], c='red', s=4)
plt.title(f'Espaço de taxas de acerto de {chr(952)}(ângulo de elevação)', fontsize = fontsize)
plt.show()


sources_number = len(r_5cm) + len(r_20cm) + len(r_1m) + len(r_2m) + len(r_error)
colors = ['lime', 'greenyellow', 'yellow', 'darkorange', 'red']

# Pie chart of r
labels = ['R<5cm', '5cm<R<20cm', '20cm<R<1m', '1m<R<2m', 'R>2m']
explode = [0, 0, 0, 0, 0]
sizes = [len(r_5cm), len(r_20cm), len(r_1m), len(r_2m), len(r_error)]
sizes = [i * (100/sources_number) for i in sizes]
fig1, ax1 = plt.subplots(figsize = (6.5,6.5))
ax1.pie(sizes, explode=explode, shadow=True, startangle=90, colors=colors)
plt.title('Taxas de acerto de R(raio)', fontsize = fontsize)
l=[f"{sizes[i]:.2f}%  {labels[i]}"for i in range(len(sizes))]
ax1.legend(l, loc=1) 
plt.tight_layout() 
plt.show()

# Pie chart of phi
labels = [f'{chr(934)}<1{chr(176)}', f'1{chr(176)}<{chr(934)}<5{chr(176)}', f'5{chr(176)}<{chr(934)}<10{chr(176)}', f'10{chr(176)}<{chr(934)}<20{chr(176)}', f'{chr(934)}>20{chr(176)}']
explode = [0, 0, 0, 0, 0]
sizes = [len(phi_1), len(phi_5), len(phi_10), len(phi_20), len(phi_error)]
sizes = [i * (100/sources_number) for i in sizes]
fig1, ax1 = plt.subplots(figsize = (6.5,6.5))
ax1.pie(sizes, explode=explode, shadow=True, startangle=90, colors=colors, rotatelabels=True)
plt.title(f'Taxas de acerto de {chr(934)}(ângulo de azimute)', fontsize = fontsize)
l=[f"{sizes[i]:.2f}%  {labels[i]}"for i in range(len(sizes))]
ax1.legend(l, loc=1) 
plt.tight_layout() 
plt.show()

# Pie chart of theta
labels = [f'{chr(952)}<1{chr(176)}', f'1{chr(176)}<{chr(952)}<5{chr(176)}', f'5{chr(176)}<{chr(952)}<10{chr(176)}', f'10{chr(176)}<{chr(952)}<20{chr(176)}', f'{chr(952)}>20{chr(176)}']
explode = [0, 0, 0, 0, 0]
sizes = [len(theta_1), len(theta_5), len(theta_10), len(theta_20), len(theta_error)]
sizes = [i * (100/sources_number) for i in sizes]
fig1, ax1 = plt.subplots(figsize = (6.5,6.5))
ax1.pie(sizes, explode=explode,shadow=True, startangle=90, colors=colors, rotatelabels=True)
plt.title(f'Taxas de acerto de {chr(952)}(ângulo de elevação)', fontsize = fontsize)
l=[f"{sizes[i]:.2f}%  {labels[i]}"for i in range(len(sizes))]
ax1.legend(l, loc=1) 
plt.tight_layout() 
plt.show()


print("Taxa de Amostragem: " + str(sampleRate) + "\n")
