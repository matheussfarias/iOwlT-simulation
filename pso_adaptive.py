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
import random
from classes import *


ini = time.time()

now = datetime.datetime.now()
date = now.strftime("%Y%m%d-%H%M%S")

if not os.path.exists("bestscores"):
    os.makedirs("bestscores")

arquivo = open("bestscores/" + date + ".txt", "a")

receptorsPositions_start= []
sampleRate = 8000
samples = int(sampleRate/2)

print("Initial Position:\n")
arquivo.write("Initial Position:\n")

    
receptorsPositions = np.zeros((4,3))



temp = receptorsPositions.copy()
receptorsPositions_start.append(temp)


print(receptorsPositions_start)
arquivo.write(np.array_str(np.array(receptorsPositions_start))+"\n\n")
start= len(receptorsPositions_start)


maxit = 100
population_quant = 10
individuos = 20
wmax = 0.9
wmin = 0.4
c1 = 2
c2 = 2
w=1
wdamp = 0.99

print("Iterations: {}".format(maxit))
print("population_quant: {}".format(population_quant))
print("individuos: {}".format(individuos))
print("wmax: {} wmin: {}".format(wmax, wmin))
print("c1: {} c2: {}".format(c1,c2))
print("sampleRate: {}".format(sampleRate))

arquivo.write("Iterations: {}\n".format(maxit))
arquivo.write("population_quant: {}\n".format(population_quant))
arquivo.write("individuos: {}\n".format(individuos))
arquivo.write("wmax: {}\nwmin: {}\n".format(wmax,wmin))
arquivo.write("c1: {}\nc2: {}\n".format(c1,c2))
arquivo.write("sampleRate: {}\n\n".format(sampleRate))


globalbest = [float("inf"), 0]
bestcost= []
conta=0
pbest=[]
world = World([], [])


for i in range (0, population_quant):
    print(i)
    population = Population([], [])
    for ind in range(0,individuos):
        if(ind < start):
            receptorsPositions = receptorsPositions_start[ind]
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
        phi = np.linspace(0, 2*np.pi,24, endpoint=False)
        theta = np.linspace(0, np.pi/2, 12, endpoint=True)
        semi_sphere = itertools.product(R, phi, theta)
    
        for (R, phi, theta) in semi_sphere:

            source = sph2car(R, phi, theta) + np.random.randn(3)*0.05
            source = np.round(source, decimals=2)
        
            rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
            toa = dist(source, receptorsPositions)/soundSpeed
            tdoa = toa[0] - toa[1:,] 
            delays = np.round(tdoa*sampleRate)/sampleRate
            #delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
            result = MLE_HLS(receptorsPositions, delays, sampleRate)
            distance += (dist(result,source))**2

        
    
        distance = distance/(15*24*12)
        x = receptorsPositions.copy()
        individuo = Individuo(x, np.zeros((4,3)), distance)
        population.addIndividuo(individuo)
        if(ind==0):
            population.setBest(individuo)
            if(i==0):
                world.setBest(individuo)
            else:
                if(individuo.getCost() < world.getBest().getCost()):
                    world.setBest(individuo)
                    
                
        else:
            if(individuo.getCost() < population.getBest().getCost()):
                population.setBest(individuo)
                if(individuo.getCost() < world.getBest().getCost()):
                    world.setBest(individuo)
                    
                    
    world.addPopulation(population)


  
for it in range(0,maxit):
    for i in range(0, population_quant):
        
        if(it==0):
            r0=0
            while(r0== 0 or r0== 0.25 or r0== 0.5 or r0== 0.75):
                r0 = np.random.rand()
                r=r0
        else:
            r = 4*r*(1-r)
            
        w = r*wmin + (wmax-wmin)*it/(maxit)
        
        
        
        
        pop = world.getPopulation(i)
        new_pop= Population([],pop.getBest())

        novas_velocidades = w*np.array(pop.getVelocities()) + c1*np.multiply(np.random.rand(4,3),np.array(pop.getBest().getPosition() - pop.getPositions())) + c2*np.multiply(np.random.rand(4,3),np.array(world.getBest().getPosition() - pop.getPositions()))

        novas_posicoes = pop.getPositions() + novas_velocidades
        
        squared = novas_posicoes**2
        radius = np.sum(squared, axis=2)
        boolean_radius = (radius>1).any(axis=1)
        

        for j in range(0, individuos):
            if(boolean_radius[j]):
                updated_individuo = Individuo(novas_posicoes[j], novas_velocidades[j],float("inf"))
                new_pop.addIndividuo(updated_individuo)
            else:
                distance=0
                R = np.linspace(1,15,15)
                phi = np.linspace(0, 2*np.pi, 24, endpoint=False)
                theta = np.linspace(0, np.pi/2, 12, endpoint=True)
                semi_sphere = itertools.product(R, phi, theta)
                
                for (R, phi, theta) in semi_sphere:
                    source = sph2car(R, phi, theta) + np.random.randn(3)*0.05
                    source = np.round(source, decimals=2)
                    
                    toa = dist(source, novas_posicoes[j])/soundSpeed
                    tdoa = toa[0] - toa[1:,] 
                    delays = np.round(tdoa*sampleRate)/sampleRate
                    
                    #delays, _ = delayEstimator (rand, sampleRate, source, novas_posicoes[j], typeComb = "")
                    result = MLE_HLS(novas_posicoes[j], delays, sampleRate)
                    distance += (dist(result,source))**2
                    
                    
                        
                distance = distance/(15*24*12)
                updated_individuo = Individuo(novas_posicoes[j], novas_velocidades[j], distance)
                new_pop.addIndividuo(updated_individuo)
                
         
            if(updated_individuo.getCost()<pop.getBest().getCost()):
                new_pop.setBest(updated_individuo)
                if(updated_individuo.getCost()<world.getBest().getCost()):
                    world.setBest(updated_individuo)
                
        world.setPopulation(new_pop, i)
        
    bestcost.append(world.getBest().getCost()[0][0])
    print("Iteration: {}, Best Cost: {}".format(it, world.getBest().getCost()) )
    arquivo.write("Iteration: {}, Best Cost: {}\n".format(it, world.getBest().getCost()) )

array = world.getBest().getPosition()
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


arquivo.write("\nBest Score: {}\n".format(world.getBest().getCost()))
arquivo.write("Best Geometry:\n")
arquivo.write(np.array_str(np.array(world.getBest().getPosition()))+"\n\n")

end= time.time()
print("Best Scores per Iteration {}".format(bestcost))
arquivo.write("Best Scores per Iteration: {}\n".format(bestcost))
print('Time {} s'.format(end-ini))
arquivo.write('Time {} s\n'.format(end-ini))

arquivo.close()