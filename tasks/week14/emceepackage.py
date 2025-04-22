#!/usr/bin/env python
# coding: utf-8

# In[1]:


import emcee
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize
import emcee
from numpy import random
import corner
import math
#from IPython.display import display, Math

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
    x_bins_values = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
    std = stda
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

# LB i also couldn't get it to work with my functions so edited the ones from the tutorial
def log_likelihood(theta, x, y):
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
    x = np.arange(0.5,10,1)
    m, b = theta
    model = m * x + b
    E = model
    v = np.array(vara)
    chi_sq = ((O-E)**2)/v 
    st = 2*math.pi*v
    secondterm = np.log(st)
    return -0.5*np.sum(chi_sq+secondterm)


def log_prior(theta):
    m, b = theta
    if 0<b<8:
        return 0
    else:
        return -np.inf


def log_probability(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)




if __name__ == "__main__":
    MH_MCMCchain(3,4,500,0.35)
    compute_lnpp(3,4)


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

    x_bins_values = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
    std = stda


    # In[ ]:





    # In[2]:


    print(compute_lnpp(3,4))
    x = np.arange(0.5,10,1)
    y = O
    yerr = stda
    m_true = 3
    b_true = 4
    slopes, inters, post, likelihood, acceptance_rate = MH_MCMCchain(3, 4, 5000, 0.35)

    mli = np.argmin(likelihood)
    bestm = slopes[mli]
    bestb = slopes[mli]
    plt.figure(figsize=(16,9))
    plt.plot(x, bestm * x + bestb, label='mcmc values')
    plt.errorbar(x_bins_values, O, yerr=std, fmt='o', color='black', label='average y')

    soln = [bestm,bestb]

    pos = soln + 1e-4 * np.random.randn(32,2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y))
    sampler.run_mcmc(pos, 5000, progress=True);


    # In[3]:


    # LB showing what the sampler did
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["m", "b"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], color='indigo', alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");


    # In[4]:


    # LB estimate integrated autocorrelation time
    tau = sampler.get_autocorr_time()
    print(tau)


    # In[5]:


    # LB flattening list of samples and throwing away initial 100 steps
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)


    # In[6]:


    # LB lotting corner plot
    fig = corner.corner(
        flat_samples, labels=labels, truths=[m_true, b_true]
    )
    fig.show();


    # In[7]:


    # Lb plot projection of results with observed data
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(x, np.dot(np.vander(x, 2), sample[:2]), color='deeppink', alpha=0.1)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(x, m_true * x + b_true, "k", label="truth")
    plt.legend(fontsize=14)
    plt.xlim(0, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show();


    # In[8]:


    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        print(txt)


    # In[ ]:




