#!/usr/bin/env python
# coding: utf-8

# In[118]:


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


# LB takes input of RA in hms and outputs x, y, z, and 1
def RAvector4arr(RA):
    c = SkyCoord(ra = RA*u.hourangle, dec = 0*u.degree, frame = 'icrs')
    RAdeg = c.ra.deg
    RAn = RAdeg + 90
    c1 = SkyCoord(ra = RAn*u.degree, dec = 0*u.degree, frame = 'icrs')
    c1.representation_type = "cartesian"
    h1 = 1
    x = c1.x.value
    y = c1.y.value
    z = c1.z.value
    return f'{x}, {y}, {z}, {h1}'



# LB takes input in Dec in degrees and outputs x, y, z, and 1-sin(dec)
def Decvector4arr(Dec):
    c1 = SkyCoord(ra = 0*u.degree, dec = 90*u.degree, frame='icrs')
    c1.representation_type = "cartesian"
    Decrad = Dec * (np.pi/180)
    h1 = 1 - math.sin(Decrad)
    x = c1.x.value
    y = c1.y.value
    z = c1.z.value
    return f'{x}, {y}, {z}, {h1}'
    


# LB takes input of RA in hms, Dec in degrees, and theta in degrees and outputs x,y,z,and 1-cos(theta)
def RADecvector4arr(RA, Dec,theta):
    c1 = SkyCoord(ra = RA*u.hourangle, dec = Dec*u.degree, frame = 'icrs')
    c1.representation_type = "cartesian"
    Decrad = Dec * (math.pi/180)
    thetarad = theta * (math.pi/180)
    h1 = 1 - math.cos(thetarad)#math.sin(Decrad)
    x = c1.x.value
    y = c1.y.value
    z = c1.z.value
    return f'{x}, {y}, {z}, {h1}'


# LB writing to a file 
def writetopolygon(RA, Dec, theta):
    file = open('sphericalcaps.txt','w')
    with open('sphericalcaps.txt', 'w') as file:
        
        file.write(f'''1 polygons
        polygon 1 ( 3 caps, 1 weight, 0 pixel, 0 str):
        {RAvector4arr(RA)}
        {Decvector4arr(Dec)}
        {RADecvector4arr(RA,Dec,theta)}''')
        
# LB testing functions
RAvector4arr(5)
Decvector4arr(36)
RADecvector4arr(5,36,1)


# In[ ]:




