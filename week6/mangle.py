#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pymangle
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


def RADecvector4arr(RA, RAunit, Dec, Decunit, theta):
    # LB making a function to take RA and Dec in any unit since i didn't last time
    '''RA and Dec in formats that can be taken by numpy, RA and Dec units in astropy units (eg u.degree), theta in degrees'''
    c1 = SkyCoord(ra = RA*RAunit, dec = Dec*Decunit, frame = 'icrs')
    c1.representation_type = "cartesian"
    Decrad = Dec * (math.pi/180)
    thetarad = theta * (math.pi/180)
    h1 = 1 - math.cos(thetarad)#math.sin(Decrad)
    x = c1.x.value
    y = c1.y.value
    z = c1.z.value
    return f'{x} {y} {z} {h1}'


def flipRADecvector4arr(RA, RAunit, Dec, Decunit, theta):
    # LB function that does the same as above but returns the negative of the bound instead of the positive
    c1 = SkyCoord(ra = RA*RAunit, dec = Dec*Decunit, frame = 'icrs')
    c1.representation_type = "cartesian"
    Decrad = Dec * (math.pi/180)
    thetarad = theta * (math.pi/180)
    h1 = -(1 - math.cos(thetarad))#math.sin(Decrad)
    x = c1.x.value
    y = c1.y.value
    z = c1.z.value
    return f'{x} {y} {z} {h1}'


def writetopolygon(RA, RA1, RAunit, RAunit1, Dec, Dec1, Decunit, Decunit1, theta, theta1, filename):
    '''RA and Dec in formats that can be taken by numpy, RA and Dec units in astropy units (eg u.degree), theta in degrees, filename should be a string'''
    # LB writes two caps to a file that is suitable for pymangle
    file = open(filename,'w')
    with open(filename, 'w') as file:
        file.write('1 polygons\n') 
        file.write('polygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
        file.write(f' {RADecvector4arr(RA,RAunit,Dec,Decunit,theta)}\n')
        file.write(f' {RADecvector4arr(RA1,RAunit1,Dec1,Decunit1,theta1)}\n')
        
def writetopolygons(RA, RA1, RAunit, RAunit1, Dec, Dec1, Decunit, Decunit1, theta, theta1, filename):
    # LB writes output of the caps made in previous functions to a file with two polygons
    file = open(filename,'w')
    with open(filename, 'w') as file: 
        file.write('1 polygons\n') 
        file.write('polygon 1 ( 1 caps, 1 weight, 0 pixel, 0 str):\n')
        file.write(f' {RADecvector4arr(RA,RAunit,Dec,Decunit,theta)}\n')
        file.write('polygon 2 ( 1 caps, 1 weight, 0 pixel, 0 str):\n')
        file.write(f' {RADecvector4arr(RA1,RAunit1,Dec1,Decunit1,theta1)}\n')

        
def flipwritetopolygon(RA, RA1, RAunit, RAunit1, Dec, Dec1, Decunit, Decunit1, theta, theta1, filename):
    # LB writes output of the caps made in previous functions to a file, flipping the sign on the constraint on cap 1
    file = open(filename,'w')
    with open(filename, 'w') as file:
        file.write('1 polygons\n') 
        file.write('polygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
        file.write(f' {flipRADecvector4arr(RA,RAunit,Dec,Decunit,theta)}\n')
        file.write(f' {RADecvector4arr(RA1,RAunit1,Dec1,Decunit1,theta1)}\n')
        

def flip2writetopolygon(RA, RA1, RAunit, RAunit1, Dec, Dec1, Decunit, Decunit1, theta, theta1, filename):
    # LB writes output of the caps made in previous functions to a file, flipping the sign on the constraint on cap 2
    file = open(filename,'w')
    with open(filename, 'w') as file: 
        file.write('1 polygons\n') 
        file.write('polygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
        file.write(f' {RADecvector4arr(RA,RAunit,Dec,Decunit,theta)}\n')
        file.write(f' {flipRADecvector4arr(RA1,RAunit1,Dec1,Decunit1,theta1)}\n')
        
        
def flipbothwritetopolygon(RA, RA1, RAunit, RAunit1, Dec, Dec1, Decunit, Decunit1, theta, theta1, filename):
    # LB writes output of the caps made in previous functions to a file, flipping the sign on the constraint on caps 1 and 2    
    file = open(filename,'w')
    with open(filename, 'w') as file:
        file.write('1 polygons\n') 
        file.write('polygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
        file.write(f' {flipRADecvector4arr(RA,RAunit,Dec,Decunit,theta)}\n')
        file.write(f' {flipRADecvector4arr(RA1,RAunit1,Dec1,Decunit1,theta1)}\n')       
        
# LB running the functions to save files for all the necessary components
writetopolygon(76,75,u.degree,u.degree,36,35,u.degree,u.degree,5,5,'intersection.ply')
writetopolygons(76,75,u.degree,u.degree,36,35,u.degree,u.degree,5,5,'bothcaps.ply')
flipwritetopolygon(76,75,u.degree,u.degree,36,35,u.degree,u.degree,5,5,'flipintersection.ply')
flip2writetopolygon(76,75,u.degree,u.degree,36,35,u.degree,u.degree,5,5,'flip2intersection.ply')
flipbothwritetopolygon(76,75,u.degree,u.degree,36,35,u.degree,u.degree,5,5,'flipbothintersection.ply')

# LB Reading in the masks using pymangle
minter = pymangle.Mangle("intersection.ply")
minterflip = pymangle.Mangle("flipintersection.ply")
minterflip2 = pymangle.Mangle("flip2intersection.ply")
minterflipboth = pymangle.Mangle("flipbothintersection.ply")
mboth = pymangle.Mangle("bothcaps.ply")

# LB generating random points for each 
ra_rand, dec_rand = minter.genrand(1000)
ra_rand1, dec_rand1 = mboth.genrand(1000)
ra_rand2, dec_rand2 = minterflip.genrand(1000)
ra_rand3, dec_rand3 = minterflip2.genrand(1000)
ra_rand4, dec_rand4 = minterflipboth.genrand(1000000)

# LB plotting the first two masks on the same plot
plt.figure(figsize=(16,9))
plt.scatter(ra_rand, dec_rand, marker = '*', s=100, color = 'hotpink', label='minter')
plt.scatter(ra_rand1, dec_rand1, marker = '*', s=100, color = 'darkseagreen', label= 'mboth')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.legend()
plt.show()

# LB plotting the second two masks on the same plot
plt.figure(figsize=(16,9))
plt.scatter(ra_rand, dec_rand, marker = '*', s=100, color = 'hotpink', label='minter')
plt.scatter(ra_rand2, dec_rand2, marker = '*', s=100, color = 'royalblue', label= 'flipped cap 1')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.legend()
plt.show()

# LB plotting the third pair of masks on the same plot
plt.figure(figsize=(16,9))
plt.scatter(ra_rand, dec_rand, marker = '*', s=100, color = 'hotpink', label='minter')
plt.scatter(ra_rand2, dec_rand2, marker = '*', s=100, color = 'royalblue', label= 'flipped cap 1')
plt.scatter(ra_rand3, dec_rand3, marker = '*', s=100, color = 'orange', label= 'flipped cap 2')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.legend()
plt.show()

# LB plotting the final mask
plt.figure(figsize=(16,9))
plt.scatter(ra_rand4, dec_rand4, marker = '*', s=100, color = 'mediumblue', label='both caps neg')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.legend()
plt.show()

# LB Flipping the signs allows us to see the portions of the cap not covered by both, but only covered by one mask.
#Flipping both covers area that is not in either mask


# In[ ]:




