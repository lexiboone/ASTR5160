#!/usr/bin/env python
# coding: utf-8

# In[62]:


from astropy.table import Table
from astropy.table import QTable
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import dustmaps
from dustmaps.config import config
from numpy.random import random
import healpy as hp
import os
import math
import pandas as pd 

# LB reading in the csv file using pandas and calling the columns we will use
file = 'week8class15.csv'
#headers = ['ObjID','ra','dec','u','g','r','i','z']
df = pd.read_csv(file, skiprows=1)
RA = df.iloc[:,1]
Dec = df.iloc[:,2]
g = df.iloc[:,4]

# LB plotting RA and Dec
plt.figure(figsize=(17,10))
plt.scatter(RA,Dec,marker='o',color='magenta')
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')

# LB defining size to be inversey proportional to g + some offset Xander showed me this
size = 3**(1/g*100-2)

# LB plotting RA and Dec with the sizes of the marker corresponding to the g, at smaller g the points will be larger
plt.figure(figsize=(17,10))
plt.scatter(RA,Dec,s=size,marker='o',color='magenta')
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
plt.show()


# In[ ]:




