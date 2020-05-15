import numpy as np
from multilateration_algorithms import *

quantity = 5
population = []
  
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


def initialization(quantity):
    population = []
  
    for i in range (0, quantity):
        r = np.random.rand(1)
        while(r==0):
            r = np.random.rand(1)
    	
        elevation = np.pi*np.random.rand(1)
        azimuthe= 2*np.pi*np.random.rand(1)
    
        mic = sph2car(r, azimuthe, elevation)
        population.append(mic)
  	
    return population
    

def calculate_fitness(receptorsPositions, source, population, samples):
    fitness = []
    for individuo in population:
        receptorsPositions[0]=np.reshape(individuo, (3,))
        
        rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1
        delays, _ = delayEstimator (rand, sampleRate, source, receptorsPositions, typeComb = "")
        result = MLE_HLS(receptorsPositions, delays, sampleRate)
        distance = float(dist(result, source))
        fitness.append(distance)

    return fitness

def takeSecond(elem):
    return elem[1]

def sortPopulation(population, fitness):
    zip_population = zip(fitness, population)
    sorted_population = sorted(zip_population)
    return sorted_population

def CrossingOver(sorted_population):
    first = sorted_population[0][1]
    second = sorted_population[1][1]
    
    direct = second- first
    t = np.random.rand(len(sorted_population))
    
    mutate_list = np.random.rand(len(sorted_population))>0.8
    newPopulation = [direct*point + first +mutate_list[i]*np.random.randn(1)*0.05 for i, point in enumerate(t)]
    return newPopulation
    #mutate_list[i]*np.random.randn(1)*0.05

sampleRate = 32000
samples = int(sampleRate/2)
    
population = initialization(quantity)
receptorsPositions = np.zeros((4,3))
receptorsPositions[0] = np.array([1, 1, 0])
receptorsPositions[1] = np.array([-1, 1, 0])
receptorsPositions[2] = np.array([1, -1, 0])
receptorsPositions[3] = np.array([-1, -1, 0])

source = np.array([5,-2, 3])

fitness = [0.1]
generation = 0
while min(fitness)>0.01:
    fitness = calculate_fitness(receptorsPositions, source, population, samples)
    sorted_population = sortPopulation(population, fitness)
    best = sorted_population[0][1]
    if generation == 0:
        bestini = best
        
    print('O melhor individuo da geracao {}: "{}"'.format(generation, best))
    print(sorted_population[0][0])
    population = CrossingOver(sorted_population)
    generation += 1





