#!/usr/bin/env python
# coding: utf-8

# In[163]:


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
#import dustmaps
import healpy as hp
import os
import math
import matplotlib.patches as patches
#import cmapy
import random
from pprint import pprint
import argparse

# LB cycling through matplotlib colors thank you Josh for this code
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def deg_to_rad(deg):
    """Retrieve Radian units from a degree input

    Parameters
    ----------
    deg : :class:`float`
        the number in degrees you want to convert to radians

    Returns
    -------
    :class:`float` 
        1 float of the radians the degrees is equivalent to
    """
    # LB converting degrees to radians
    return deg * (math.pi/180)

def lat_lon_area(RAmin, RAmax, Decmin, Decmax):
    """Retrieve the lat-lon area of the rectangle bound by two RA and Dec measurements

    Parameters
    ----------
    RAmin : :class:`float`
        the RA min you want to use in degrees
    RAmax : :class:`float`
        the RA max you want to use in degrees
    Decmin : :class:`float`
        the Dec min you want to use in degrees
    Decmax : :class:`float`
        the Dec max you want to use in degrees

    Returns
    -------
    :class:`float` 
        1 float of the area of the rectangle bound by the RAs and Decs inputted
    """
    # LB calculating the area of the rectangle in square degrees
    area = (180/math.pi)*(RAmax - RAmin)*(math.sin(Decmax*(math.pi/180)) - math.sin(Decmin*(math.pi/180)))
    return area

def make_rectangle(RAmin, RAmax, Decmin, Decmax, color):
    """Retrieve the lat-lon area of the rectangle bound by two RA and Dec measurements

    Parameters
    ----------
    RAmin : :class:`float`
        the RA min you want to use in degrees
    RAmax : :class:`float`
        the RA max you want to use in degrees
    Decmin : :class:`float`
        the Dec min you want to use in degrees
    Decmax : :class:`float`
        the Dec max you want to use in degrees

    Returns
    -------
    :class:`matplotlib.patches.Rectangle` 
        A rectangle that is bound by the RA min, RAmax, Decmin, Decmax
    """
    # LB finding the area of the lat-lon rectangle and putting it into a string
    areaf = f'{lat_lon_area(RAmin, RAmax, Decmin, Decmax):.2f}'
    area = str(areaf) + ' square degrees'
    print(area)
    # LB Making the rectangle into a matplotlib patch 
    patch = patches.Rectangle((deg_to_rad(RAmin), deg_to_rad(Decmin)), (math.pi/180)*(RAmax-RAmin), (math.pi/180)*(Decmax-Decmin), edgecolor=cycle[color], facecolor='white', linewidth=2, label=area)
    return patch    
    
def plotting(RAmin, RAmax, Decmin, Decmax, filepath):
    """Plot the rectangle in the aitoff projection

    Parameters
    ----------
    RAmin : :class:`float`
        the RA min you want to use in degrees
    RAmax : :class:`float`
        the RA max you want to use in degrees
    Decmin : :class:`float`
        the Dec min you want to use in degrees
    Decmax : :class:`float`
        the Dec max you want to use in degrees
    filepath : :class:`str`
    the file path you want to save the figure to


    Returns
    -------
    A plot in the aitoff projection with rectangles that are bound by the RA min, RAmax, Decmin, Decmax
    """
    # LB setting figure size and putting it into the aitoff projection
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="aitoff")
    # LB making rectangles in the same RA but increasing Dec 
    rect1 = make_rectangle(RAmin, RAmax, Decmin, Decmax, 0)
    rect2 = make_rectangle(RAmin, RAmax, Decmin+17, Decmax+32, 1)
    rect3 = make_rectangle(RAmin, RAmax, Decmin+35, Decmax+50, 2)
    rect4 = make_rectangle(RAmin, RAmax, Decmin+70, Decmax+85, 3)
    # LB Adding those rectangles to a patch on the plot 
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.legend()
    ax.grid(color='b', linestyle='dashed', linewidth=1)
    # LB saving figure in the file path specified
    plt.savefig(os.path.join(filepath, 'lat_lon_rectangles.png'))

def random_pop(RAmin, RAmax, Decmin, Decmax, filepath):
    """Plot random points in a sphere and plot the points inside a rectangle bound by RA min, RA max, Dec min, and Dec max in a different color

    Parameters
    ----------
    RAmin : :class:`float`
        the RA min you want to use in degrees
    RAmax : :class:`float`
        the RA max you want to use in degrees
    Decmin : :class:`float`
        the Dec min you want to use in degrees
    Decmax : :class:`float`
        the Dec max you want to use in degrees
    filepath : :class:`str`
        the file path you want to save the figure to

    Returns
    -------
    A plot in the aitoff projection of 100000 random RA and Dec points, and the points inside the rectangle plotted in a different color
    """
    # LB making figure size and putting it into the aitoff projection
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="aitoff") 
    # LB generating 100000 points of RA and Dec
    ragen = 2 * np.pi * (np.random.random(100000) - 0.5)
    decgen = np.arcsin(1 - np.random.random(100000) * 2.)
    # LB indexing only the points in the box bound by the RA min, RA max, Dec min, and Dec max
    i = (ragen >= deg_to_rad(RAmin)) & (ragen <= deg_to_rad(RAmax)) & (decgen >= deg_to_rad(Decmin)) & (decgen <= deg_to_rad(Decmax))
    # LB printing the ratio of points within the rectangle to the points in the total area to compare to the area difference
    print(len(ragen[i])/len(ragen))
    # LB plotting the random points and overplotting the points in the box in a different color
    ax.scatter(ragen,decgen, color='purple',label='Random points', s=1, marker='*')
    ax.scatter(ragen[i],decgen[i], color='dodgerblue',label='In the lat-lon projection', s=1, marker='*')
    ax.legend()
    ax.grid(color='b', linestyle='dashed', linewidth=1)
    # LB saving figure in the file path specified
    plt.savefig(os.path.join(filepath, 'inside_rect.png'))
    print(ragen[i],decgen[i])
    return (ragen[i],decgen[i])

if __name__ == "__main__":
    # LB testing the functions with 0,30,0,15.
    lat_lon_area(0,30,0,15)
    plotting(0,30,0,15,'.')
    random_pop(0,30,0,15,'.')
    # LB comparing to see if the area fraction matches with the point fractions
    fraction = lat_lon_area(0,30,0,15)/(4*np.pi*180*180/np.pi/np.pi)
    print(fraction)

parser = argparse.ArgumentParser()
# LB allowing parser so we can input the file path we want to save figures to
parser.add_argument("filepath", type=str, help="Where would you like to save your figure?")
args = parser.parse_args()
FILEPATH = args.filepath

plotting(0,30,0,15,FILEPATH)
random_pop(0,30,0,15,FILEPATH)



# In[ ]:





# In[ ]:




