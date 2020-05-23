#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:33:41 2020

@author: matheus
"""

import numpy as np
from multilateration_algorithms import *
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits import mplot3d
import time
import os
import datetime


ini = time.time()

now = datetime.datetime.now()
date = now.strftime("%Y%m%d-%H%M%S")

if not os.path.exists("bestscores"):
    os.makedirs("bestscores")

arquivo = open("bestscores/" + date + ".txt", "a")

sampleRate = 4000
samples = int(sampleRate/2)

print("Initial Position:\n")
arquivo.write("Initial Position:\n")

    
receptorsPositions = np.zeros((4,3))
receptorsPositions[0] = np.array([0.42154551, -0.77801156,  0.07249975])
receptorsPositions[1] = np.array([-0.04487197, -0.02304995,  0.38198168])
receptorsPositions[2] = np.array([0.00635872, -0.54369001, -0.30237201])
receptorsPositions[3] = np.array([0.62097167, -0.13719707, -0.23648541])

receptorsPositions_start = receptorsPositions
print(receptorsPositions_start)
arquivo.write(np.array_str(receptorsPositions_start)+"\n\n")
'''
[16.877140107102075,
 array([[-0.18872155, -0.17237658,  0.76836726],
        [-0.04487197, -0.02304995,  0.38198168],
        [ 0.00635872, -0.54369001, -0.30237201],
        [ 0.62097167, -0.13719707, -0.23648541]])]


[273.30581585621985,
 array([[ 0.42154551, -0.77801156,  0.07249975],
        [-0.08168635,  0.81548885, -0.43835588],
        [-0.1145777 , -0.08060338, -0.31211295],
        [-0.6041687 , -0.6674966 , -0.24889886]])]


receptorsPositions[0] = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[1] = np.array([np.sqrt(2)/2, -np.sqrt(2)/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0])
'''

maxit = 3
quantity = 3
w = 1
c1 = 2
c2 = 2
wdamp = 0.99

print("Iterations: {}".format(maxit))
print("quantity: {}".format(quantity))
print("w: {}".format(w))
print("c1: {} c2: {}".format(c1,c2))
print("wdamp: {}".format(wdamp))
print("sampleRate: {}".format(sampleRate))

arquivo.write("Iterations: {}\n".format(maxit))
arquivo.write("quantity: {}\n".format(quantity))
arquivo.write("w: {}\n".format(w))
arquivo.write("c1: {}\nc2: {}\n".format(c1,c2))
arquivo.write("wdamp: {}\n".format(wdamp))
arquivo.write("sampleRate: {}\n\n".format(sampleRate))

population = []
globalbest = [float("inf"), 0]
bestcost= []
conta=0


for i in range (0, quantity):
    print(i)
    if(i == 0):
        receptorsPositions = receptorsPositions_start
    else:
        for j in range(0,4):
            r = np.random.rand(1)
            while(r==0 or r>1):
                r = np.random.rand(1)
	
            elevation = np.pi*np.random.rand(1)
            azimuthe= 2*np.pi*np.random.rand(1)
            receptorsPositions[j] = sph2car(r, azimuthe, elevation).reshape((1,3))
    
    
    distance=0
    R = np.linspace(1,15,6)
    phi = np.linspace(0, 2*np.pi,6, endpoint=False)
    theta = np.linspace(0, np.pi/2, 6, endpoint=True)
    semi_sphere = itertools.product(R, phi, theta)
    
    for (R, phi, theta) in semi_sphere:

        source = sph2car(R, phi, theta) + np.random.randn(3)*0.1
        source = np.round(source, decimals=2)
        
        rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
        delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
        result = MLE_HLS(receptorsPositions, delays, sampleRate)
        distance += float(dist(result, source))
    
    x = receptorsPositions.copy()
    population.append([x, distance, np.zeros((4,3)), x, distance])
    
    if population[i][4] < globalbest[0]:
        globalbest[0] = population[i][4]
        globalbest[1] = population[i][3]
      
      
for it in range(0,maxit):
    for i in range (1, quantity):
        population[i][2] = w*population[i][2] + np.random.rand(4,3)*c1*(population[i][3]-population[i][0]) + c2*np.random.rand(4,3)*(globalbest[1] - population[i][0])
        
        population[i][0] = population[i][0] + population[i][2]
        
        r1 = car2sph(population[i][0][0][0],population[i][0][0][1],population[i][0][0][2])
        r2 = car2sph(population[i][0][1][0],population[i][0][1][1],population[i][0][1][2])
        r3 = car2sph(population[i][0][2][0],population[i][0][2][1],population[i][0][2][2])
        r4 = car2sph(population[i][0][3][0],population[i][0][3][1],population[i][0][3][2])
        
        if(r1[0]>1 or r2[0]>1 or r3[0]>1 or r4[0]>1):
            population[i][1] = float("inf")
        else:
            receptorsPositions = population[i][0]
            R = np.linspace(1,15,6)
            phi = np.linspace(0, 2*np.pi,6, endpoint=False)
            theta = np.linspace(0, np.pi/2, 6, endpoint=True)
    
            semi_sphere = itertools.product(R, phi, theta)
            distance=0
            for (R, phi, theta) in semi_sphere:
                source = sph2car(R, phi, theta) + np.random.randn(3)*0.1
                source = np.round(source, decimals=2)
                
                delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
                result = MLE_HLS(receptorsPositions, delays, sampleRate)
                distance += float(dist(result, source))
                
            population[i][1] = distance
            
        
        
        if population[i][1] < population[i][4]:
            population[i][3] = population[i][0]
            population[i][4] = population[i][1]
            
            
            if population[i][4] < globalbest[0]:
                globalbest[0] = population[i][4]
                globalbest[1] = population[i][3]
                
    bestcost.append(globalbest[0])
    print("Iteration: {}, Best Cost: {}".format(it, globalbest[0]) )
    arquivo.write("Iteration: {}, Best Cost: {}\n".format(it, globalbest[0]) )
    w=w*wdamp
    

receptorsPositions[3] = globalbest[1][0]

array = receptorsPositions
array_ax = plt.axes(projection='3d')
array_ax.set_xlim([-1.5, 1.5])
array_ax.set_xlim([-1.5, 1.5])
array_ax.set_xlim([-1.5, 1.5])
array_ax.scatter3D(array[:,0], array[:,1], array[:,2], c='darkblue', depthshade=False , s=40)
for i in range(array.shape[0]):
    array_ax.text(array[i,0], array[i,1], array[i,2]+0.1, f"mic {str(i)}", zorder=1)
plt.title("Disposição espacial dos microfones", fontsize=12)
plt.show()

plt.plot(bestcost)
plt.show()

print("Best Score: {}".format(globalbest[0]))
print("Best Geometry:")
print(globalbest[1])

arquivo.write("\nBest Score: {}\n".format(globalbest[0]))
arquivo.write("Best Geometry:\n")
arquivo.write(np.array_str(globalbest[1])+"\n\n")

end= time.time()
print('Time {} s'.format(end-ini))
arquivo.write('Time {} s\n'.format(end-ini))
arquivo.close()
    