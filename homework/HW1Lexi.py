#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


def quasar(month):
    """Find the lowest airmass quasars that can be observed in a month

    Keyword arguments:
    month- the month in a number 1-12
    """
    # LB  defining the quasar directory to be the directory /d/scratch/ASTR5160/week4 in .bashrc file or directing to the right one if there isn't one set
    quasardir = os.getenv("QUASARDIR")
    if quasardir is None:
        print("Set the QUASARDIR environment variable to point to the main " +
        "QUASARDIR data directory (e.g. /d/scratch/ASTR5160/week4)")
    # LB reading in the file 
    filename = 'HW1quasarfile.txt'
    file = open(os.path.join(quasardir, filename), "r")  
    df = np.genfromtxt(file, dtype=str)
    
    # Finding RA and Dec in formats astropy will take
    RA = [f'{i[:2]} {i[2:4]} {i[4:9]}' for i in df]
    Dec = [f'{i[9:12]} {i[12:14]} {i[14:18]}' for i in df]

    # LB Setting the positions of these RA,Dec pairs in SkyCoords
    c = SkyCoord(ra = RA, dec = Dec, unit=(u.hourangle,u.deg))

    # LB finding RA and Dec in degrees
    RAdeg = c.ra.deg
    Decdeg = c.dec.deg
    
    # LB finding the number of days given what month we are in in this year
    _, number_of_days = calendar.monthrange(2025, month)
    
    # LB Making an array of all the days in the month and formatting it so single digits will have a 0 in front
    days = np.arange(1,number_of_days+1)
    newdays = ['%02d' % float(x) for x in days]
     
    # LB Finding the offset so we can transform our times from MST to UTC 
    utcoffset = -7 * u.hour 

    # LB setting date of every day in the month at 11pm in a string
    date = [f'2025-{month:02d}-{x} 23:00:00' for x in newdays]

    # LB converting the string back into astropy time attribute and converting to UTC
    astropytime = astropy.time.core.Time(date) - utcoffset

    # LB setting KPNO to be the location of kitt peak observatory
    KPNO = EarthLocation.of_site('kpno')
    
    # LB printing table headers
    print("Date (MST)              ","Quasar Coordinates ", "RA (◦) ", "Dec (◦) ", "Airmass")
    print("                        ","(hms.ss ◦ ′ ′′)", "     ", "      ", "       ")

    # LB looping over every day in the month 
    for i in astropytime:
        # LB finding the airmasses of each quasar at each day in the month
        aa = c.transform_to(AltAz(location = KPNO, obstime = i))
        airmass = aa.secz
        # LB Finding only positive airmass
        d = (airmass > 0)
        posairmass = airmass[d]
        # Finding the position of the minimum positive airmass and finding the corresponding index of the quasar position and RA and Dec in degrees
        pos = np.argmin(posairmass)
        minairmass = posairmass[pos]
        quasar = df[pos]
        # LB converting time back into MST
        backtomst = str(i+utcoffset)
        RAt = RAdeg[pos]
        Dect = Decdeg[pos]
        # LB adding these values to the table for every day in the month and printing table
        table = [[f'{backtomst}',f'{quasar}',f'{RAt}',f'{Dect}',f'{minairmass}']] 
        print(tabulate(table))
        

parser = argparse.ArgumentParser()
# LB allowing parser so we can just input the month used in the function
parser.add_argument("month", type=int, help="What month of the year would you like to observe in? (Between 1-12)")
args = parser.parse_args()

MONTH = args.month

quasar(MONTH)




