# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:50:46 2020

@author: davim, matheussfarias
"""

import numpy as np
from itertools import combinations as comb
import scipy.signal as scc

soundSpeed = 340.29

def dist (Pi, Pj) : 
    """ Calculate the distance of the arrays Pi and Pj
       
        Pi, Pj : 1D or 2D ndarrays , each line is a coordinate vector 
        
        return 2D ndarray with the distance/distances between each line of Pi and Pj 
        
        Broadcasting allowed
    """
    i = 1 if (Pi.ndim == 2 or Pj.ndim == 2) else None 
    diff = Pi - Pj
    square = diff**2
    dist = np.sqrt(np.sum(square, axis=i, keepdims=True))
    dist = dist.reshape((dist.shape[0], 1)) if (i == None) else dist
    return dist

def signal (rand, delay, sampleRate, multError = 1, sumError = 0):
    """
        rand : Random vector between 1 and -1 to help the gunshot generator (defined in main)
        delay : float with the delay between the funcSignal and the received signal
        sampleRate : sampling rate of the receptors
        multError : gain of the receptor sensor
        sumError : standard deviation of the normal noise received by the receptor
        
        
        return 1D ndarray with shape (sampleRate/2, ) that represents the delayed signal received
    """
    assert (delay<0.5 and delay>0)
    samples = int(sampleRate/2)
    time = np.linspace(0, 0.5, samples)
    noise = np.random.randn(samples)
    return multError * create_shot(time - delay, samples, rand, 0) + sumError * noise

def signals (rand, delays, sampleRate, multErrors = [], sumErrors = [], echo_remove_boolean = 0):
    """
        rand : Random vector between 1 and -1 to help the gunshot generator (defined in main)
        delays : 2D column ndarray with the delays between the funcSignal and the received signal in each receptor
        sampleRate : sampling rate of the receptors
        multErrors : 1D ndarray with gains of each receptor sensor
        sumError : 1D ndarray with standard deviation of the normal noise received by each receptor
        
        
        return 2D ndarray with shape (numberOfReceptors, sampleRate/2) that represents the delayed signal received
        by each receptor
    """
    l = delays.shape[0]
    samples = int(sampleRate/2)
    signals = np.zeros((l, samples))
    multE = np.ones(l) if type(multErrors) != np.ndarray else multErrors
    sumE = np.zeros(l) if type(sumErrors) != np.ndarray else sumErrors
    for i in range(l):
        signals[i] = signal(rand, delays[i,0], sampleRate, multE[i], sumE[i])
        if(echo_remove_boolean):
            alpha, N = find_echo_coef(signals[i])
            signals[i] = echo_remove(signals[i], alpha, N)
    return signals
    
def tal (signal1, signal2, sampleRate):
    """
        sinal1, signal2 : 1D ndarray with the two delayed signals
        sampleRate : sampling rate of the receptors
        
        return : float with the delay time between signal1 and signal2
    """
    samples = int(sampleRate/2)
    corr = scc.correlate(signal1, signal2, "full")
    delay = (corr.argmax()-(samples-1))/sampleRate
    return delay

def delayEstimator (rand, sampleRate, sourcePosition, receptorsPositions, multErrors = [], sumErrors = [], typeComb = "all comb"):
    """ 
        rand : Random vector between 1 and -1 to help the gunshot generator (defined in main)
        sampleRate : sampling rate of the receptors
        sourcePosition : 1D ndarray with the coordinates of the source of the signal emmited 
        receptorsPositions : row 2D ndarray with the coordinates of all receptors
        
        return : column 2D ndarray with the estimated delays between the combinations of the signal received by
        each receptor
    """
    realDelays = dist(sourcePosition, receptorsPositions) / soundSpeed
    l = receptorsPositions.shape[0]
    
    sounds = signals (rand, realDelays, sampleRate, multErrors, sumErrors)  
    
    estimatedDelays = []
    if typeComb == "all comb":
        for i,j in comb(range(l), 2):
            estimatedDelays.append(tal(sounds[i], sounds[j], sampleRate))
    else:
        for i in range(1,l):
            estimatedDelays.append(tal(sounds[0], sounds[i], sampleRate))
    return np.array(estimatedDelays, ndmin=2).T, sounds

def mleMatrices (delays, receptorsPositions):
    """ Generates matrices for MLE(Maximum likelihood estimation) calculation
        
        delays : column 2D ndarray with the TDOA from some reference receptor
        receptorsPositions : row 2D ndarray with the coordinates of all receptors
        
        return matrices D,E that are used in the MLE calculation
        
        Xs = D * D1 + E
        
        Xs => [xs ys zs].T estimate source coordinates column vector
        D1 => estimated distance between the source and the reference receptor
    """
    
    B = -delays * soundSpeed
    C = B**2
    A = receptorsPositions[1:] - receptorsPositions[0]
    Caux = -receptorsPositions[1:]**2 + receptorsPositions[0]**2
    Caux = np.sum(Caux, axis=1, keepdims=True)
    C = 0.5 * (C + Caux)
    
    A = -np.linalg.pinv(A)
    D = np.dot(A, B)
    E = np.dot(A, C)
    
    return D,E

def possibleCoords (A, B, referencePosition):
    """ Calculates the two possible coordinates for the given MLE problem
        
        A : D matrix from mleMatrices function
        B : E matrix from mleMatrices function
        
        return a row 2D ndarray with the two possible coordinates for the given MLE multilateration solution
    """
    a = np.sum(A**2) - 1
    b = 2 * ( np.sum(A * B) - np.sum(A * referencePosition) )
    c = np.sum(B**2) - 2 * np.sum(B * referencePosition) + np.sum(referencePosition ** 2)
    coeffs = [a, b, c]
    roots = np.roots(coeffs).reshape(2,1)
    finalCoords = (np.dot(A,roots.T)+B).T
    return finalCoords

def MLE(receptorsPositions, delays, sampleRate):
    """ MLE(Maximum likelihood estimation) algorithm
        
        sampleRate : sampling rate of the receptors
        delays : column 2D ndarray with the TDOA from some reference receptor
        receptorsPositions : row 2D ndarray with the coordinates of all receptors

        
        return a row 2D ndarray with the two possible coordinates for the given MLE problem
    """
    A, B = mleMatrices (delays, receptorsPositions)
    result = possibleCoords (A, B, receptorsPositions[0].reshape((-1,1)))
    return result


def MLE_HLS(receptorsPositions, delays, sampleRate, typeComb=''):
    m,_ = receptorsPositions.shape
    results = np.real( MLE(receptorsPositions, delays[:(m-1),], sampleRate) )########
    costs = cost(results, delays, receptorsPositions, typeComb=typeComb)
    i = np.argmin(costs)
    return results[i]

def h(r1, r2, guess):
    """ Calculate the difference of the distances between r1, guess and r2, guess
        r1, r2 and guess : 1D or 2D row ndarray with coordinates
        
        return 1D ndarray with the difference of the distances between r1, guess and r2, guess
    """
    return (dist(r1, guess) - dist(r2, guess))

def dh(r1, r2, guess):
    dist_r1 = dist(r1, guess)
    dist_r2 = dist(r2, guess)
    deriv = (r1 - guess)/dist_r1 - (r2 - guess)/dist_r2 
    if np.ndim(guess) == 1: return deriv.reshape(-1)   
    else: return deriv

def cost(guess, delays, receptorsPositions, typeComb = "all comb"):
    """ Calculate the sum of the squares of the loss function of hyperbolic least squares problem
    
        guess : 1D or 2D row ndarray with one or more guesses coordinates
        delays : column 2D ndarray with the TDOA from combinations of all receptors
        receptorsPositions : row 2D ndarray with the coordinates of all receptors       
        
        return 2D ndarray of shape (number_of_guesses, 1) with the costs related to all guesses
        
        loss_ij = delays_ij * soundSpeed - h(receptorsPositions_i, receptorsPositions_j, guess)
        
        cost = sum in combinations of ij (loss_ij ** 2)        
    """
    l = receptorsPositions.shape[0]
    m =  1 if np.ndim(guess) == 1 else guess.shape[0] 
    cost = np.zeros((m, 1))
    calcDists = delays * soundSpeed
    if typeComb == "all comb":
        for x,y in enumerate( comb(range(l), 2) ):
            (i, j) = (y[0], y[1])
            r1 = receptorsPositions[i]
            r2 = receptorsPositions[j]
            cost += np.abs(calcDists[x] - h(r1, r2, guess))**2
    else:
        r1 = receptorsPositions[0]
        for i,j in enumerate( range(1,l) ):
            r2 = receptorsPositions[j]
            cost += np.abs(calcDists[i] - h(r1, r2, guess))**2
    return cost

def create_shot (t,samples, rand, echo_boolean=0, echo_gain = np.random.uniform(0.1,0.3)):
    """ Create a gunshot sound with/without echo
    
        t : 1D linspace array
            example:
                    t = np.linspace(0, 0.5, samples)
                    
        samples : sampling rate divided by 2
        
        rand : Random vector between 1 and -1 to help the gunshot generator (defined in main)
        
        echo_boolean : 1, if want the gunshot to has echo
                       0, if want the gunshot not to has echo
                       
        echo_gain : the echo gain (between 0 and 1)
            
    """
    
    decayConst = -10
    relu = (t>=0)
    index = np.sum(~relu)
    r = np.append(np.zeros(index), rand[0:rand.shape[0]-index])
    r = r * np.exp(decayConst * t) * relu
    
    if(echo_boolean):
        phase = int(np.random.uniform(samples/4,samples/1.5))
        impulse = scc.unit_impulse(samples, phase)
        echo = scc.convolve(r,impulse)
        echo = echo[:samples]
        r += echo*echo_gain
    
    return r

def find_echo_coef(signal):
    """ Finds the echo coefficients alpha and N of the equation
        y[n] = x[n] + \alpha . x[n-N], where x[n] is the original gunshot sound without echo.
        
    
        signal : the y[n] signal
            
    """
    R = scc.correlate(signal, signal, mode = 'full')
    centre_index = np.argmax(R)
    phase_index = np.argmax(R[centre_index+1:]) + centre_index+1
    R_centre = R[centre_index]
    R_phase = R[phase_index]
    coef = -R_centre/R_phase
    p = [1, coef, 1]
    roots = np.roots(p)
    alpha = min(roots)
    N = phase_index - centre_index
    return alpha, N

def echo_remove(signal, alpha, N):
    """ Returns the x[n] signal of the equation below
        y[n] = x[n] + \alpha . x[n-N], where x[n] is the original gunshot sound without echo.
        
    
        signal : the y[n] signal
        
        alpha : the \alpha coefficient
        
        N : the N coefficient
            
    """
    a = np.zeros(N+1)
    a[0]= 1
    a[-1] = alpha
    a = np.array(a)
    result = scc.lfilter(np.array([1]),a,signal)
    return result



############################## AS FUNÇÕES ABAIXO ESTÃO INCOMPLETAS ######################################
def GausNewtonActualization(receptorsPositions, guess):
    fact = np.math.factorial
    l = receptorsPositions.shape[0]
    n = receptorsPositions.shape[1]
    m = int( fact(l)/(fact(l-2) * 2) ) 
    J = np.zeros((m, n))
    r = np.zeros((m,1))
    for x,y in enumerate( comb(range(l), 2) ):
        (i, j) = (y[0], y[1])
        r1 = receptorsPositions[i]
        r2 = receptorsPositions[j]
        J[x] = dh(r1, r2, guess)
        r[x] = h(r1, r2, guess)
    A = np.dot( np.linalg.pinv(J), r ).reshape(-1)
    return A

def HLE(delays, sampleRate, receptorsPositions, numIterations = 100):
    guess = np.zeros(receptorsPositions[0].shape)
    for i in range(numIterations):
        A = GausNewtonActualization(receptorsPositions, guess)
        guess += A
        print(guess)
    return guess
    