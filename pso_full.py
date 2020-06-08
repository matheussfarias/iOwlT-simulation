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

receptorsPositions_start= []
sampleRate = 16000
samples = int(sampleRate/2)

print("Initial Position:\n")
arquivo.write("Initial Position:\n")

    
receptorsPositions = np.zeros((4,3))
receptorsPositions[0] = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[1] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([-0.2030, -0.02199, 0.24133])

temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)

#square
receptorsPositions[0] = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[1] = np.array([np.sqrt(2)/2, -np.sqrt(2)/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0])

temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)

#pyramid
receptorsPositions[0] = np.array([0, 1, 0])
receptorsPositions[1] = np.array([-np.sqrt(3)/2, -1/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([0, 0, 1])

temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)

#square 1-shifted
receptorsPositions[0] = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[1] = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(3)/2])

temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)

#trapezoid
receptorsPositions[0] = np.array([1/2, np.sqrt(3)/2, 0])
receptorsPositions[1] = np.array([-1/2, np.sqrt(3)/2, 0])
receptorsPositions[2] = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2, 0])
receptorsPositions[3] = np.array([np.sqrt(3)/4, -1/4, np.sqrt(3)/2])

temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)

#best until now
receptorsPositions[0] = np.array([-0.36268539, -0.47297163, -0.23418507])
receptorsPositions[1] = np.array([0.86186511,  0.16530619,  0.28758431])
receptorsPositions[2] = np.array([0.22206284,  0.21997007, -0.1301347])
receptorsPositions[3] = np.array([-0.04290313,  0.98427498, -0.15466009])

temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)


print(receptorsPositions_start)
arquivo.write(np.array_str(np.array(receptorsPositions_start))+"\n\n")
start= len(receptorsPositions_start)


maxit = 100
quantity = 10
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
    if(i < start):
        receptorsPositions = receptorsPositions_start[i]
    else:
        for j in range(0,4):
            r = np.random.rand(1)
            while(r==0 or r>1):
                r = np.random.rand(1)
	
            elevation = np.pi*np.random.rand(1)
            azimuthe= 2*np.pi*np.random.rand(1)
            receptorsPositions[j] = sph2car(r, azimuthe, elevation).reshape((1,3))
    
    
    distance=0
    R = np.linspace(1,15,15)
    phi = np.linspace(0, 2*np.pi,12, endpoint=False)
    theta = np.linspace(0, np.pi/2, 6, endpoint=True)
    semi_sphere = itertools.product(R, phi, theta)
    
    for (R, phi, theta) in semi_sphere:

        source = sph2car(R, phi, theta) + np.random.randn(3)*0.05
        source = np.round(source, decimals=2)
        
        rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
        delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
        result = MLE_HLS(receptorsPositions, delays, sampleRate)
        distance += (dist(result,source))**2

        
    
    distance = distance/(15*12*6)
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
            R = np.linspace(1,15,15)
            phi = np.linspace(0, 2*np.pi, 12, endpoint=False)
            theta = np.linspace(0, np.pi/2, 6, endpoint=True)
    
            semi_sphere = itertools.product(R, phi, theta)
            distance=0
            for (R, phi, theta) in semi_sphere:
                source = sph2car(R, phi, theta) + np.random.randn(3)*0.05
                source = np.round(source, decimals=2)
                
                delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
                result = MLE_HLS(receptorsPositions, delays, sampleRate)
                distance += (dist(result,source))**2
            
            distance = distance/(15*12*6)
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
arquivo.write(np.array_str(np.array(globalbest[1]))+"\n\n")

end= time.time()
print("Best Scores per Iteration {}".format(bestcost))
arquivo.write("Best Scores per Iteration: {}\n".format(bestcost))
print('Time {} s'.format(end-ini))
arquivo.write('Time {} s\n'.format(end-ini))

arquivo.close()
    