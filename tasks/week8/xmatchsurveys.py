#!/usr/bin/env python
# coding: utf-8

# In[58]:


import astropy
from astropy.table import Table
from astropy.table import QTable
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from tabulate import tabulate
from astropy.time import Time
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import calendar
import argparse
import os
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

# LB reading in and opening the first radio data using astropy Table
os.environ['SDSSDIR']= '/d/scratch/ASTR5160/data/first'
sdssdir = os.getenv("SDSSDIR")
filename = 'first_08jul16.fits'
file = os.path.join(sdssdir, filename)
objs = Table.read(file)
RA = objs["RA"]
Dec = objs["DEC"]

# LB plotting RA and Dec
plt.figure(figsize=(16,9))
plt.scatter(RA,Dec,marker='*')
plt.show()

# LB indexing first 100 data points of the first radio data
RAh = RA[0:100]
Dech = Dec[0:100]
bothh = np.column_stack((RAh,Dech))
 
# LB obtaining SDSS optical data for first 100 objects in first radio data
os.system("python sdssDR9query.py RAh Dech >> week8file.txt")

# LB finding all the sweep file names in the directory
user_dir = '/d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep'
file_list = []
for dirpath,_,filenames in  os.walk(user_dir):
    for name in filenames:
        if name.endswith('.fits'):
            file_list.append(os.path.join(user_dir,dirpath,name))
#print(file_list)
# LB using DESI code and editing for this use
def decode_sweep_name(sweepname, nside=None, inclusive=True, fact=4):
    """Retrieve RA/Dec edges from a full directory path to a sweep file

    Parameters
    ----------
    sweepname : :class:`str`
        Full path to a sweep file, e.g., /a/b/c/sweep-350m005-360p005.fits
    nside : :class:`int`, optional, defaults to None
        (NESTED) HEALPixel nside
    inclusive : :class:`book`, optional, defaults to ``True``
        see documentation for `healpy.query_polygon()`
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_polygon()`

    Returns
    -------
    :class:`list` (if nside is None)
        A 4-entry list of the edges of the region covered by the sweeps file
        in the form [RAmin, RAmax, DECmin, DECmax]
        For the above example this would be [350., 360., -5., 5.]
    :class:`list` (if nside is not None)
        A list of HEALPixels that touch the  files at the passed `nside`
        For the above example this would be [16, 17, 18, 19]
    """
    # ADM extract just the file part of the name.
    sweepname = os.path.basename(sweepname)

    # ADM the RA/Dec edges.
    ramin, ramax = float(sweepname[6:9]), float(sweepname[14:17])
    decmin, decmax = float(sweepname[10:13]), float(sweepname[18:21])

    # ADM flip the signs on the DECs, if needed.
    if sweepname[9] == 'm':
        decmin *= -1
    if sweepname[17] == 'm':
        decmax *= -1

    if nside is None:
        return [ramin, ramax, decmin, decmax]

    pixnum = hp_in_box(nside, [ramin, ramax, decmin, decmax],
                       inclusive=inclusive, fact=fact)

    return pixnum

def is_in_box(objs, radecbox):
    """Determine which of an array of objects are inside an RA, Dec box.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        An array of objects. Must include at least the columns "RA" and "DEC".
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax] forming the
        edges of a box in RA/Dec (degrees).

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in the box, ``False`` for objects outside of the box.

    Notes
    -----
        - Tests the minimum RA/Dec with >= and the maximum with <
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM check for some common mistakes.
    if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
        msg = "Strange input: [ramin, ramax, decmin, decmax] = {}".format(radecbox)
        log.critical(msg)
        raise ValueError(msg)

    ii = ((objs[:,0] >= ramin) & (objs[:,0] < ramax)
          & (objs[:,1] >= decmin) & (objs[:,1] < decmax))
    #print(objs[0])
    return ii

# LB seeing which files have any objects fall within it. It will print them and save them into sweepfiles.
sweepfiles = []
def findsweep(objs,files):
    for i in files:
        ramin, ramax, decmin, decmax = decode_sweep_name(i)
        radecbox1 = ramin, ramax, decmin, decmax 
        if np.any(is_in_box(objs,radecbox1)) == True:
            print(i)
            sweepfiles.append(i)
     
    
    
findsweep(bothh,file_list)
print(sweepfiles)
#I am sorry I only did red tasks this week I got behind on work but I will practice with the stuff more.


# In[ ]:




