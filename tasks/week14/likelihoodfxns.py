#!/usr/bin/env python
# coding: utf-8

# In[13]:


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy import random
import math
# LB function to calculate ln posterior probability
def compute_lnlikelihood(m,b):
    """Calculates ln posterior probability

    Parameters
    ----------
    m,b: float (slope and intercept)
    
    Returns
    -------
    ln posterior probability for the function given in '/d/scratch/ASTR5160/week13/line.data'
    given some m and b. 
    
    Notes
    -------
    Will return -np.inf for b less than 0 or larger than 8 (values are outside of m,b space we 
    want to sample)
    """
    x = np.arange(0.5,10,1)
    yact = O
    E = x*m + b
    v = np.array(vara)
    chi_sq = ((O-E)**2)/v 
    st = 2*math.pi*v
    secondterm = np.log(st)
    return -0.5*np.sum(chi_sq+secondterm) #+ np.log(1)

    #print(y)
    
def compute_lnpp(m,b):
    """Calculates ln posterior probability

    Parameters
    ----------
    m,b: float (slope and intercept)
    
    Returns
    -------
    ln posterior probability for the function given in '/d/scratch/ASTR5160/week13/line.data'
    given some m and b. 
    
    Notes
    -------
    Will return -np.inf for b less than 0 or larger than 8 (values are outside of m,b space we 
    want to sample)
    """
    x = np.arange(0.5,10,1)
    yact = O
    E = x*m + b
    v = np.array(vara)
    chi_sq = ((O-E)**2)/v 
    st = 2*math.pi*v
    secondterm = np.log(st)
    if 0<b<8:
        return -0.5*np.sum(chi_sq+secondterm) + np.log(1)
    else:
        return -np.inf
    #print(y)
    

# def MH_MCMCchain(m,b):
#     """Calculates R and returns new values of m and b according to gaussian proposal function

#     Parameters
#     ----------
#     m,b: float (slope and intercept as priors)
    
#     Returns
#     -------
#     ln posterior probability - ln new posterior probability for the function given 
#     in '/d/scratch/ASTR5160/week13/line.data' given some m and b and a step size. 
    
#     Notes
#     -------
#     Will tell you when you should accept new parameters based on if R >1
#     """
#     delta_m = 0.1
#     delta_b = 0.1
#     # LB gaussian propoasl function with steps defined above
#     newm = np.random.normal(m,delta_m)
#     newb = np.random.normal(b,delta_b)
#     lnR = compute_lnpp(m,b) - compute_lnpp(newm,newb)
#     if np.exp(lnR) > 1:
#         print('Always accept new parameters')
#         return np.exp(lnR), newm, newb
        
#     else:
#         print(f'Accept new parameters with probability {np.exp(lnR)}')
#         return np.exp(lnR), newm, newb
        

def MH_MCMCchain(m, b, iterations, stepsize):
    
    # Modified from eliza's code because I got stuck and she helped me

    x = np.arange(0.5,10,1)
    y = m * x + b

    slopes = [m]
    intercepts = [b]
    posterior = [compute_lnpp(m,b)]
    likelihoods = [compute_lnlikelihood(m,b)]

    accepted = 0

    for i in range(iterations):
        # New slope based on some step size

        updatedm = random.normal(slopes[-1], stepsize)
        updatedb = random.normal(intercepts[-1], stepsize)


        # LB find lnR
        lnR = compute_lnpp(updatedm,updatedb) - compute_lnpp(m,b)
        R = np.exp(lnR)

        if R > 1:
            # LB always accept
            slopes.append(updatedm)
            intercepts.append(updatedb)
            posterior.append(compute_lnpp(updatedm, updatedb))
            likelihoods.append(compute_lnlikelihood(updatedm, updatedb))
            accepted += 1
        else:
            # LB - if R < 1, generate a random number between 0 and 1
            rand_number = random.random()

            if R < rand_number:
                # LB - if R is less than random number, reject these new parameters & move to next iteration
                continue
            elif R > rand_number:
                # LB - if R is greater than the random number accept these new parameters
                slopes.append(updatedm)
                intercepts.append(updatedb)
                posterior.append(compute_lnpp(updatedm, updatedb))
                likelihoods.append(compute_lnlikelihood(updatedm, updatedb))
                accepted += 1

    acceptance_rate = (accepted / iterations) * 100

    return slopes, intercepts, posterior, likelihoods, round(acceptance_rate, 3)

if __name__ == "__main__":
    file = glob.glob('/d/scratch/ASTR5160/week13/line.data')[0]

    # Reading in file
    df = pd.read_csv(file, delim_whitespace=True) # For tab-separated data
    #print(df)
    #print(df[0])
    # print(df.columns)

    x0 = np.array([df['#x']])
    # print(x0)
    x1 = np.array([df['0.5']])
    x2 = np.array([df['1.5']])
    x3 = np.array([df['2.5']])
    x4 = np.array([df['3.5']])
    x5 = np.array([df['4.5']])
    x6 = np.array([df['5.5']])
    x7 = np.array([df['6.5']])
    x8 = np.array([df['7.5']])
    x9 = np.array([df['8.5']])

    x_bins = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
    x_bins_names = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']
    O = []
    vara = []
    stda =[]
    # finding mean and variance and std
    for i in range(len(x_bins)):
        mean = np.mean(x_bins[i])
        var = np.var(x_bins[i], ddof = 1)
        std = np.std(x_bins[i], ddof=1)
        O.append(mean)
        vara.append(var)
        stda.append(std)
        print(f'The mean for {x_bins_names[i]} is {mean}')
        print(f'The variance for {x_bins_names[i]} is {var}')


    print(vara)


    # In[ ]:





    # In[14]:


    MH_MCMCchain(3,4,500,0.35)
    compute_lnpp(3,4)


    # In[ ]:





