#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:10:04 2020

@author: matheus
"""
import numpy as np

class Individuo:
    def __init__(self, position, velocity, cost):
        self.position = position
        self.velocity = velocity
        self.cost = cost
        
    def setPosition(self, position):
        self.position = position
        
    def setVelocity(self, velocity):
        self.velocity = velocity
        
    def setCost(self, cost, index=0):
        self.cost = cost
        
    def getPosition(self):
        return self.position
    
    def getVelocity(self):
        return self.velocity
        
    def getCost(self):
        return self.cost
    
class Population:
    def __init__(self, individuos, best):
        self.individuos = individuos
        self.best = best
        
    def setBest(self, best):
        self.best = best
        
    def addIndividuo(self, individuo):
        self.individuos.append(individuo)
    
    def getBest(self):
        if (self.best == []):
            x = Individuo([],[],float("inf"))
            return x
        else:
            return self.best
    
    def getPosition_individuo(self,i):
        return self.individuos[i].getPosition()
    
    def getPositions(self):
        positions = []
        for i in range(0,len(self.individuos)):
            positions.append(self.individuos[i].getPosition())
        return positions
    
    def getVelocities(self):
        velocities = []
        for i in range(0,len(self.individuos)):
            velocities.append(self.individuos[i].getVelocity())
        return velocities
    
    def changeVelocities(self):
        self.individuos
    
    def getCosts(self):
        costs = []
        for i in range(0,len(self.individuos)):
            costs.append(self.individuos[i].getCost())
        return costs
    
    def setCost(self,cost,index):
        self.individuos[index].cost = cost 
    
class World:
    def __init__(self, populations, globalbest):
        self.populations = populations
        self.globalbest = globalbest
        
    def setBest(self, globalbest):
        self.globalbest = globalbest
        
    def addPopulation(self, population):
        self.populations.append(population)
    
    def getBest(self):
        return self.globalbest
    
    def getPopulation(self,i):
        return self.populations[i]
    
    def setPopulation(self, population, index):
        self.populations[index] = population
    