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

population = []  


sampleRate = 32000
samples = int(sampleRate/2)
    
receptorsPositions = np.zeros((4,3))
receptorsPositions[0] = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[1] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])



maxit = 100
quantity = 3
w = 1
c1 = 2
c2 = 2
wdamp = 0.99

population = []
globalbest = [float("inf"), 0]
bestcost= []
conta=0
for i in range (0, quantity):
    print(i)
    r = np.random.rand(1)
    while(r==0 or r>1):
        r = np.random.rand(1)
	
    elevation = np.pi*np.random.rand(1)
    azimuthe= 2*np.pi*np.random.rand(1)
    mic = sph2car(r, azimuthe, elevation).reshape((1,3))

    receptorsPositions[3] = mic
    distance=0
    R = np.linspace(1,15,15)
    phi = np.linspace(0, 2*np.pi,24, endpoint=False)
    theta = np.linspace(0, np.pi/2, 12, endpoint=True)
    
    semi_sphere = itertools.product(R, phi, theta)

    for (R, phi, theta) in semi_sphere:

        source = sph2car(R, phi, theta) + np.random.randn(3)*0.1
        source = np.round(source, decimals=2)
        
        rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
        delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
        result = MLE_HLS(receptorsPositions, delays, sampleRate)
        distance += float(dist(result, source))
          
    population.append([mic, distance, np.zeros((1,3)), mic, distance])
        
    if population[i][4] < globalbest[0]:
        globalbest[0] = population[i][4]
        globalbest[1] = population[i][3]
      
      
for it in range(0,maxit):
    for i in range (1, quantity):
        population[i][2] = w*population[i][2] + np.random.rand(1,3)*c1*(population[i][3]-population[i][0]) + c2*np.random.rand(1,3)*(globalbest[1] - population[i][0])
        
        population[i][0] = population[i][0] + population[i][2]
        
        r = car2sph(population[i][0][0][0],population[i][0][0][1],population[i][0][0][2])
        if(r[0]>1):
            population[i][1] = float("inf")
        else:
            receptorsPositions[3] = population[i][0]
            R = np.linspace(1,15,15)
            phi = np.linspace(0, 2*np.pi,24, endpoint=False)
            theta = np.linspace(0, np.pi/2, 12, endpoint=True)
    
            semi_sphere = itertools.product(R, phi, theta)
            distance=0
            for (R, phi, theta) in semi_sphere:
                delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
                result = MLE_HLS(receptorsPositions, delays, sampleRate)
                distance += float(dist(result, source))
                
            population[i][1] = distance
            
        if( (population[i][0]==receptorsPositions[0]).all() or (population[i][0]==receptorsPositions[1]).all() or (population[i][0]==receptorsPositions[2]).all() or (population[i][0]==source).all()):
            population[i][1] = float("inf")  
        
        
        
        if population[i][1] < population[i][4]:
            population[i][3] = population[i][0]
            population[i][4] = population[i][1]
            
            
            if population[i][4] < globalbest[0]:
                globalbest[0] = population[i][4]
                globalbest[1] = population[i][3]
                
    bestcost.append(globalbest[0])
    print("Iteration: {}, Best Cost: {}".format(it, globalbest[0]) )
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

    