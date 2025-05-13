#!/usr/bin/env python
# coding: utf-8

# In[12]:


import emcee
import corner
import numpy as np
import glob 
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize
import math 
#from IPython.display import display, Math
import argparse
from argparse import ArgumentParser

# LB all functions below were used from the emcee tutorial fitting a model to data https://emcee.readthedocs.io/en/stable/tutorials/line/

def log_likelihood_linear(theta, x, y, yerr):
    """Finds the log of the likelihood function of a line
    
    Parameters
    ----------
    theta : :class:'array'
        array containing m and b values you want to compute the likeilihood function for
        example: [m  b  ]
    x, y, yerr : :class:'array'
        arrays containing x, y, and standard deviation values you want to compute the likeilihood function for (The raw data)

    Returns
    -------
    The log of the likelihood function of a line fitting some raw data at values of m and b
    
    """
    
    # LB calling our parameters we want to fit theta and calling the model the straight line fit
    m, b = theta
    model = m * x + b
    v = yerr**2
    
    # LB computing chi squared
    chi_sq = ((y - model)**2) / v 

    # LB computing the second term in the log likelihood from eqn 1 of class notes
    st = 2*math.pi*v
    secondterm = np.log(st)

    # LB returning the log of the likelihood function
    return -0.5*np.sum(chi_sq+secondterm)

def log_prior_linear(theta):
    """Finds the log of the prior function of a line
    
    Parameters
    ----------
    theta : :class:'array'
        array containing m and b values you want to compute the likeilihood function for
        example: [m  b  ]
    Returns
    -------
    The log of the priors for values of m and b

    Notes
    -------
    Uses flat or uninformative priors, will return 0 for values within the bounds we expect and -infinity outside those bounds
    
    """

    # LB reweighting the likelihood based on priors, priors were found manually by finding bad fits to data
    # LB using a flat prior
    m, b = theta
    if -2.0 < m < -0.3 and -7 < b < 9:
        return 0.0
    return -np.inf
    
def log_probability_linear(theta, x, y, yerr):
    """Finds the log of the probability function of a line
    
    Parameters
    ----------
    theta : :class:'array'
        array containing m and b values you want to compute the likeilihood function for
        example: [m  b  ]
    x, y, yerr : :class:'array'
        arrays containing x, y, and standard deviation values you want to compute the likeilihood function for (The raw data)

    Returns
    -------
    The log of the probability for values of m and b

    Notes
    -------
    Adds the log likelihood and log prior functions together
    
    """
    
    # LB combining the prior and likelihood to form probability function
    lp = log_prior_linear(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_linear(theta, x, y, yerr)


def log_likelihood_quad(theta, x, y, yerr):
    """Finds the log of the likelihood function of a quadratic function
    
    Parameters
    ----------
    theta : :class:'array'
        array containing coefficients you want to compute the likelihood function for
        example: [a2 a1  a0]
    x, y, yerr : :class:'array'
        arrays containing x, y, and standard deviation values you want to compute the likeilihood function for (The raw data)

    Returns
    -------
    The log of the likelihood function of a quadratic fitting some raw data at values of a2, a1, a0
    
    """

    # LB calling our parameters we want to fit theta and calling the model the quadratic fit
    a2, a1, a0 = theta
    model = a2 * x**2 + a1 * x + a0
    v = yerr**2

    # LB computing chi squared
    chi_sq = ((y - model)**2) / v 
    st = 2*math.pi*v

    # LB computing the second term in the log likelihood from eqn 1 of class notes
    secondterm = np.log(st)
    return -0.5*np.sum(chi_sq+secondterm)

def log_prior_quad(theta):
    """Finds the log of the prior function of a quadratic function
    
    Parameters
    ----------
    theta : :class:'array'
        array containing coefficients you want to compute the likelihood function for
        example: [a2 a1  a0]
    Returns
    -------
    The log of the priors for values of the coefficients

    Notes
    -------
    Uses flat or uninformative priors, will return 0 for values within the bounds we expect and -infinity outside those bounds
    
    """
    
    # LB reweighting the likelihood based on priors, priors were found manually by finding bad fits to data
    # LB using a flat prior
    a2, a1, a0 = theta
    if -2.0 < a2 < 2 and -7 < a1 < 7 and -15 < a0 < 17:
        return 0.0
    return -np.inf
    
def log_probability_quad(theta, x, y, yerr):
    """Finds the log of the probability function of a line
    
    Parameters
    ----------
    theta : :class:'array'
        array containing m and b values you want to compute the likeilihood function for
        example: [m  b  ]
    x, y, yerr : :class:'array'
        arrays containing x, y, and standard deviation values you want to compute the likeilihood function for (The raw data)

    Returns
    -------
    The log of the probability for values of m and b

    Notes
    -------
    Adds the log likelihood and log prior functions together
    
    """
    # LB combining the prior and likelihood to form probability function
    lp = log_prior_quad(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_quad(theta, x, y, yerr)

        
def fitting_emcee_linear(filename, m_true, b_true, steps):
    """Finds the best fit of a line using emcee
    
    Parameters
    ----------
    filename : :class:'str'
        The full path to the file that contains the data you want to use in fits file format.
        Data should be in format (x, y, yerr)
    m_true, b_true : :class:'float'
        values of m and b you want to be the starting guess for the parameters
    steps : :class:'float'
        Number of steps you want to use with emcee 
        
    Returns
    -------
    A corner plot of the parameter's posterior probability distributions
    A plot of the data with the model represented by the best-fitting parameters
    Prints the best fitting parameters and uncertainties found by emcee
    
    Notes
    -------
    If the corner plots still do not look like Gaussians, try again with more steps
    
    """

    print('Running MCMC for fitting linear model to data')
    # LB Reading in file and assigning according variables
    file = glob.glob(filename)[0]
    data = fits.open(file)[1].data
    x, y, yerr = np.array([[row[0], row[1], row[2]] for row in data]).T    

    # LB adding a random seed so results will be reproducable
    np.random.seed(42)

    # LB getting initial guesses by optimizing likelihood function
    nll = lambda *args: -log_likelihood_linear(*args)
    initial = np.array([m_true, b_true]) + 0.1 * np.random.randn(2)
    soln = minimize(nll, initial, args=(x, y, yerr))
    m_ml, b_ml = soln.x
    

    # LB sampling the distribution using emcee
    pos = soln.x + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability_linear, args=(x, y, yerr)
    )
    sampler.run_mcmc(pos, steps, progress=True);
    samples = sampler.get_chain()
    tau = sampler.get_autocorr_time()

    # LB Throwing away 3 times the number of steps needed to forget as burn-in and thinning by half the autocorrelation time
    flat_samples = sampler.get_chain(discard=round(max(tau)*3), thin=round(max(tau)/2), flat=True)
    

    # LB creating corner plot
    labels = ["m", "b"]
    xsmooth = np.arange(min(x)-1,max(x)+1,0.01)
    fig = corner.corner(
    flat_samples, labels=labels, color='indigo')#, truths=[m_true, b_true])
    plt.show()

    # LB plotting y(x) and best fit
    plt.figure(figsize=(16,9))
    plt.errorbar(x, y, yerr=yerr, fmt="*", color='purple', capsize=0, ms=10, label='Raw Data')

    # LB finding best fit parameters and uncertainties based on 16th, 50th, and 84th percentiles of the samples
    mcmcs = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(2)]
    (mfit, qm), (bfit, qb) = [(m[1], np.diff(m)) for m in mcmcs]

    # LB plotting smoothed line with best fitting parameters
    plt.plot(xsmooth, mfit * xsmooth + bfit, color='indigo', label=f"MCMC best fit: y = {mfit:.3f}$x$ + {bfit:.3f}")
    plt.legend(fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # LB printing best-fitting parameters and uncertainty
    print("Best fiting parameters for linear fit:")
    print(f"m = {mfit:.3f} ± ^{qm[0]:.3f}_{qm[1]:.3f}")
    print(f"b = {bfit:.3f} ± ^{qb[0]:.3f}_{qb[1]:.3f}") 


def fitting_emcee_quad(filename, a2_true, a1_true, a0_true, steps):
    """Finds the best fit of a line using emcee
    
    Parameters
    ----------
    filename : :class:'str'
        The full path to the file that contains the data you want to use in fits file format.
        Data should be in format (x, y, yerr)
    a2_true, a1_true, a0_true : :class:'float'
        values of quadratic coefficients you want to be the starting guess for the parameters
    steps : :class:'float'
        Number of steps you want to use with emcee 
        
    Returns
    -------
    A corner plot of the parameter's posterior probability distributions
    A plot of the data with the model represented by the best-fitting parameters
    Prints the best fitting parameters and uncertainties found by emcee
    
    Notes
    -------
    If the corner plots still do not look like Gaussians, try again with more steps
    
    """
    
    print('Running MCMC for fitting quadratic model to data')
    # LB Reading in file and assigning according variables
    file = glob.glob(filename)[0]
    data = fits.open(file)[1].data
    x, y, yerr = np.array([[row[0], row[1], row[2]] for row in data]).T   

    # LB adding a random seed so results will be reproducable
    np.random.seed(42)

    # LB getting initial guesses by optimizing likelihood function
    nll = lambda *args: -log_likelihood_quad(*args)
    initial = np.array([a2_true, a1_true, a0_true]) + 0.1 * np.random.randn(3)
    soln = minimize(nll, initial, args=(x, y, yerr))
    a2_ml, a1_ml, a0_ml = soln.x
    
    # LB sampling the distribution using emcee
    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape 
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability_quad, args=(x, y, yerr)
    )
    sampler.run_mcmc(pos, steps, progress=True);
    samples = sampler.get_chain()
    tau = sampler.get_autocorr_time()

    # LB Throwing away 3 times the number of steps needed to forget as burn-in and thinning by half the autocorrelation time
    flat_samples = sampler.get_chain(discard=round(max(tau)*3), thin=round(max(tau)/2), flat=True)

    # LB creating corner plot
    labels = ["a2", "a1", "a0"]
    xsmooth = np.arange(min(x)-1,max(x)+1,0.01)
    fig = corner.corner(
    flat_samples, labels=labels, color='indigo')#, truths=[m_true, b_true])
    plt.show()
    

    # LB plotting y(x) and best fit
    plt.figure(figsize=(16,9))
    plt.errorbar(x, y, yerr=yerr, fmt="*", color='purple', capsize=0 , ms=10, label='Raw Data')

    # LB finding best fit parameters and uncertainties based on 16th, 50th, and 84th percentiles of the samples
    mcmcs = [np.percentile(flat_samples[:, i], [16, 50, 84]) for i in range(3)]
    (a2fit, qa2), (a1fit, qa1), (a0fit, qa0) = [(m[1], np.diff(m)) for m in mcmcs]
    

    # LB plotting smoothed line with best fitting parameters
    plt.plot(xsmooth, a2fit * xsmooth**2 + a1fit * xsmooth + a0fit, color='indigo', label=f"MCMC best fit: y = {a2fit:.3f}$x^2$ {a1fit:.3f}$x$ + {a0fit:.3f}")
    plt.legend(fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # LB printing best-fitting parameters and uncertainty
    print("Best fiting parameters for quadratic fit:")
    print(f"a2 = {a2fit:.3f} ± ^{qa2[0]:.3f}_{qa2[1]:.3f}")
    print(f"a1 = {a1fit:.3f} ± ^{qa1[0]:.3f}_{qa1[1]:.3f}") 
    print(f"a0 = {a0fit:.3f} ± ^{qa0[0]:.3f}_{qa0[1]:.3f}")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LB adding argument of a file path to argparse so when read a filepath at the CL it can do an MCMC fit using linear and quadratic models
    ap = ArgumentParser(description='Fits a linear and quadratic line to data using the emcee package')
    args = ap.parse_args()
    #FILEPATH = args.filepath

    # LB running the functions for both the quadratic and linear fit using the file path to data
    # LB edit this if want to change starting point or number of steps
    fitting_emcee_linear('/d/scratch/ASTR5160/final/dataxy.fits', -0.7, 1, 10000)
    fitting_emcee_quad('/d/scratch/ASTR5160/final/dataxy.fits', 0.7, -2, 8, 10000)
    print('Looking at the poseriors for a2, a quadratic fit is warranted, though not by much. Since the a2 value is small, but still does not encompass 0 when accounting for error associated with the 16th and 84th percentiles, we cannot say that the quadratic fit is not needed, even though the posteriors do have some overlap with 0.')
    print('Therefore, we are justified in using a quadratic fit even though the linear fit is not too bad.')
    print('Visually, I think the quadratic fit looks to fit the data bettter.')


# In[ ]:





# In[ ]:





# In[ ]:




