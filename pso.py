#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:33:41 2020

@author: matheus
"""

import numpy as np
from multilateration_algorithms import *

quantity = 50
population = []  


sampleRate = 32000
samples = int(sampleRate/2)
    
receptorsPositions = np.zeros((4,3))
receptorsPositions[0] = np.array([1, 1, 0])
receptorsPositions[1] = np.array([-1, 1, 0])
receptorsPositions[2] = np.array([1, -1, 0])
receptorsPositions[3] = np.array([-1, -1, 0])

source = np.array([3,-2, 0])


maxit = 100
quantity = 5
w = 1
c1 = 2
c2 = 2
wdamp = 0.99

population = []
globalbest = [float("inf"), 0]
bestcost= []
  
for i in range (0, quantity):
    r = np.random.rand(1)
    while(r==0):
        r = np.random.rand(1)
	
    elevation = np.pi*np.random.rand(1)
    azimuthe= 2*np.pi*np.random.rand(1)
    mic = sph2car(r, azimuthe, elevation).reshape((1,3))

    receptorsPositions[3] = mic
    rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
    delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
    result = MLE_HLS(receptorsPositions, delays, sampleRate)
    distance = float(dist(result, source))
    population.append([mic, distance, np.zeros((1,3)), mic, distance])
    
    if population[i][4] < globalbest[0]:
        globalbest[0] = population[i][4]
        globalbest[1] = population[i][3]
      
for it in range(0,maxit):
    for i in range (1, quantity):
        population[i][2] = w*population[i][2] + np.random.rand(1,3)*c1*(population[i][3]-population[i][0]) + c2*np.random.rand(1,3)*(globalbest[1] - population[i][0])
        
        population[i][0] = population[i][0] + population[i][2]
        
        
        receptorsPositions[3] = population[i][0]
        delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
        result = MLE_HLS(receptorsPositions, delays, sampleRate)
        distance = float(dist(result, source))
        population[i][1] = distance
        
        if population[i][1] < population[i][4]:
            population[i][3] = population[i][0]
            population[i][4] = population[i][1]
            
            
            if population[i][4] < globalbest[0]:
                globalbest[0] = population[i][4]
                globalbest[1] = population[i][3]
                
    bestcost.append(globalbest[0])
    print("Iteration: {}, Best Cost: {}".format(it, globalbest[0]) )
    w=w*wdamp