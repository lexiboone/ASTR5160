#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[118]:


import astropy
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import os
import astropy.units as u
import math 
import glob
import numpy as np
from time import sleep 
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tasks.week12.flaggingdata import *
from homework.sdssDR9query import *


def findsources(RA_cap, Dec_cap, radius, objects):
    """Finds sources in a given radius 

    Parameters
    ----------
    RA_cap, Dec_cap, radius : :class:`float`
        RA, Dec, and radius of the area you want to search around. RA and Dec should be the center of the circle of the radius you want to search around.
        All should be in degrees
    objects : :class:'astropy table'
        objects in an atropy table that you want to find matches around

    Returns
    -------
    objects inside circle :class:`astropy table`
        objects within the search radius
    """
    # LB reading in the skycoords object for the center of the circle you want to search around
    c1 = SkyCoord(ra = RA_cap, dec = Dec_cap, frame = 'icrs', unit=(u.degree, u.degree))
    # LB reading in the coordinates for the objects you want to find within that circle
    c2 = SkyCoord(ra = objects["RA"], dec = objects["DEC"], frame = 'icrs', unit=(u.degree, u.degree))
    # Finding the index of which the separation is less than the desired separation radius and returning only those objects
    ind = (c1.separation(c2) < radius*u.degree)
    return objects[ind]

def mag_nanomaggies(flux):
    """Converts flux in nanomaggies to magnitude

    Parameters
    ----------
    flux : :class:`float`
        flux in nanomaggies

    Returns
    -------
    magnitude :class:`float`
        magnitude
    """
    return 22.5 - 2.5 * np.log10(flux)


def crossmatch(matching_radius, filenames, max_rband, min_wisecolors, RAc, Decc, radiusc, objs_table):
    """Cross matches objects in the FIRST and Legacy surveys.
    
    Parameters
    ----------
    matching_radius : :class:`float`
        Matching radius you want to use to match the objects in the surveys.
        All should be in arcseconds 
    filenames : :class:`list`
        A list of the Legacy Survey sweep files you want to match to the FIRST survey.
    max_rband, min_wisecolors : :class:`float`
        The maximum r and minimum W1-W2 colors you want to make your cuts around. If you want all objects with r mag < 22 your input should be 22, etc.
    RAc,Decc,radiusc : :class:`float`
        RA, Dec, and radius of the area you want to search around. RA and Dec should be the center of the circle of the radius you want to search around.
        All should be in degrees
    objs_table : :class:'astropy table'
        objects in an atropy table that you want to find matches around
    

    Returns
    -------
    Legacy and FIRST sources which match :class:`table`
        Astropy tables of the legacy and first surveys which cross matched given the matching radius.
    """

    # LB first finding sources within a certain survey in the FIRST survey (see findsources documentation above) and reading into SkyCoord
    firstsources_insurvey = findsources(RAc, Decc, radiusc, objs_table)
    RA_is = firstsources_insurvey["RA"]
    Dec_is = firstsources_insurvey["DEC"]
    firstsourcecoord = SkyCoord(RA_is, Dec_is, unit=(u.degree, u.degree))
    combinedis = np.column_stack((RA_is,Dec_is))

    # LB finding the sweepfile names that correspond to the astronomer's survey objects in the FIRST survey 
    sweepfiles_is = findsweep(firstsources_insurvey,filenames)

    # LB stacking the files to read into SkyCoord to get a large table with all the coordinates used
    totfiles = vstack([Table.read(file, unit_parse_strict = 'silent') for file in sweepfiles_is])
    totcoordinates = SkyCoord(totfiles["RA"], totfiles["DEC"], unit=(u.degree, u.degree))

    # LB finding the indices of the objects that match and only finding the unique indices. 
    id1, id2, d2, d3 = totcoordinates.search_around_sky(firstsourcecoord, u.arcsecond * matching_radius)
    id1s, uid1 = np.unique(id1, return_index=True)
    id2s = id2[uid1]
    id2s, uid2 = np.unique(id2s, return_index=True)
    id1s = id1s[uid2]

    # Finding the object matches in the sweep files and FIRST file
    matches = totfiles[id2s]
    #print(len(matches))
    FIRSTmatches = firstsources_insurvey[id1s]
    # RA_match = matches["RA"]
    # Dec_match = matches["DEC"]
    
    #objs1 = Table.read(sweepfiles_is[0], format='fits', unit_parse_strict='silent')
    #print(objs1)
    #print(objs1)

    # LB finding the fluxes of R, W1, W2 and converting to magnitude
    R_flux = matches["FLUX_R"]
    R_magnitudes =  mag_nanomaggies(R_flux)
    #R_index = R_magnitudes < 22
    W1_flux = matches["FLUX_W1"]
    W2_flux = matches["FLUX_W2"]
    W1_magnitudes =  mag_nanomaggies(W1_flux)
    W2_magnitudes =  mag_nanomaggies(W2_flux)   

    # LB finding W1-W2
    W1_W2 = W1_magnitudes-W2_magnitudes
     # LB finding only the objects in both tables with colors according to the color cutoffs we set. 
    W_index = (W1_W2 > min_wisecolors) & (R_magnitudes < max_rband)
    #r_matches = matches[R_index]
    W_matches = matches[W_index]
    firstW_matches = FIRSTmatches[W_index]
    #print(W_matches)
    # LB returning the tables of only the matches with the criteria we gave.
    return W_matches, firstW_matches

def sdss_query_modified(RA, Dec):
    """Matches RA and Dec to SDSS query objects. Modified from Dr. Myers sdssDR9query module

    Parameters
    ----------
    RA, Dec : :class:`float`
        Right Ascension and Declination to query around in SDSS DR9

    Returns
    -------
    SDSS result :class:`list`
        RA, Dec, and ugriz of the object in SDSS DR9. If no object is found, it is populated with the RA and Dec inputted and 1000 
        for each of the magnitudes.
    """
    # ADM initialize the query.
    qry = sdssQuery()
    
    # ADM the query to be executed. You can substitute any query, here!
    query = """SELECT top 1 ra,dec,u,g,r,i,z FROM PhotoObj as PT
    JOIN dbo.fGetNearbyObjEq(""" + str(RA) + """,""" + str(Dec) + """,0.02) as GNOE
    on PT.objID = GNOE.objID ORDER BY GNOE.distance"""
    
    # ADM execute the query.
    qry.query = query
    for line in qry.executeQuery():
        result = line.strip()
    
    # ADM NEVER remove this line! It won't speed up your code, it will
    # ADM merely overwhelm the SDSS server (a denial-of-service attack)!
    sleep(1)

    # ADM the server returns a byte-type string. Convert it to a string.
    # LB saving to a variable in addition to printing
    #print(result.decode())
    sdssoutput = result.decode()
    
    # LB splitting the output and saving the ones that don't have magnitudes with 1000 mag
    output_list = sdssoutput.split(',')
    #print(output_list)
    #output_list_nums = [float(n) for n in output_list]
    try:
        output_list_nums = [float(n) for n in output_list]
    except ValueError:
        output_list_nums = [RA, Dec, 1000, 1000, 1000, 1000, 1000]
    #print(output_list_nums)
    return output_list_nums


def sdss_query_modified_table(RAs, Decs):
    """Matches RA and Dec to SDSS query objects for multiple coordinates. Modified from Dr. Myers sdssDR9query module
    See also sdss_query_modified module

    Parameters
    ----------
    RAs, Decs : :class:`float`
        Right Ascensions and Declinations to query around in SDSS DR9

    Returns
    -------
    SDSS result :class:`table`
        RA, Dec, and ugriz of the object in SDSS DR9, stacked into a table so each row is the result of the query for one object.
    """
    # LB making a table that stacks the output of the SDSS query so multiple RA, Dec points can be ran through the query
    output = [sdss_query_modified(RAs[i], Decs[i]) for i in range(len(RAs))]
    table = np.row_stack(output)
    table = Table(table)
    #print(table)
    return table
def flux_nanomaggies(mag):
    """Converts magnitude to flux in nanomaggies 

    Parameters
    ----------
    magnitude : :class:`float`
        magnitude of an object

    Returns
    -------
    flux :class:`float`
        flux in nanomaggies
    """
    return 10**((-1/2.5)*(mag-22.5))


if __name__ == "__main__":
    # LB making a help message for argparse
    ap = ArgumentParser(description='Cross matches objects between the FIRST and Legacy surveys with certain color cuts. Then queries with SDSS to eventually find the brightest object in the u band, to plot flux as a function of wavelength.')
    args = ap.parse_args()
    
    # LB reading in the sweep files and the FIRST file
    sweepfilenames = glob.glob('/d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0/*.fits')
    file = glob.glob('/d/scratch/ASTR5160/data/first/first_08jul16.fits')[0]

    # LB adding the coordinates of the first file into SkyCoords
    objs_table = Table.read(file)
    RA = objs_table["RA"]
    Dec = objs_table["DEC"]
    firstobjs = SkyCoord(ra = RA, dec = Dec, frame = 'icrs', unit=(u.degree, u.degree))

    # LB finding the objects that lie in the survey
    objects_in_survey = findsources(163, 50, 3.0, objs_table)
    
    # LB cross matching the objects in the FIRST survey with the legacy survey
    sweep_table, FIRST_table = crossmatch(1,sweepfilenames,22,0.5,163,50,3,objs_table)
    print(f'The total number of Legacy survey sources that match the FIRST sources is {len(sweep_table)}.')
    
    
    # LB using sdss query to find sdss matches for the objects and their corresponding magnitudes
    colors = sdss_query_modified_table(sweep_table["RA"],sweep_table["DEC"])
    
    # LB assigning the oututs of the sdss query to the corresponding variables
    ra,dec,u,g,r,i,z = colors['col0'], colors['col1'], colors['col2'], colors['col3'], colors['col4'], colors['col5'], colors['col6']

    # LB indexing the values which sdss query found a match (non-matches had magnitudes replaced with 1000)
    match = u < 1000
    print(f'The total number of sources with a match in the SDSS database is {len(ra[match])}')

    # LB reassigning variables to the objects that did have a match
    ra,dec,u,g,r,i,z = ra[match],dec[match],u[match],g[match],r[match],i[match],z[match]
    sweep_tablem, FIRST_tablem = sweep_table[match], FIRST_table[match]
    #print(ra)

    # LB finding the index of the minimum u value (brightest object in u) and indexing the magnitudes of that object
    ubrite1_index = np.argmin(u)

    ubrite1_u = u[ubrite1_index]
    ubrite1_i = i[ubrite1_index]

    # LB converting the magnitudes to flux
    FLUX_U = flux_nanomaggies(ubrite1_u)
    FLUX_I = flux_nanomaggies(ubrite1_i)
    ui_flux = np.array([FLUX_U, FLUX_I])

    # LB finding the fluxes of the bands other than u and i for the object ubrite1
    list_names = ['FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_W3','FLUX_W4']

    fluxes_ubrite = {}
    fluxes_values = []
    for i in range(len(list_names)):
        values = sweep_tablem[list_names[i]][ubrite1_index]
        variable_name = f'{list_names[i]}'
    
        fluxes_ubrite[variable_name] = values
        fluxes_values.append(values)
    print(f'The location of ubrite1 is RA {ra[ubrite1_index]}, Dec {dec[ubrite1_index]}.')

    # LB putting all the flux values and wavelength values together for plotting purposes
    fluxes_values = np.array(fluxes_values)
    allfluxes = np.concatenate((ui_flux, fluxes_values))
    # wavlengths are u, i, g, r, z, W1, W2, W3, W4
    wavelengths_A = np.array([3543, 7625, 4770, 6231, 9134, 34000, 46000, 120000, 220000])
    ind_sortedw = np.argsort(wavelengths_A)
    
    
    # LB plotting fluxes as a function of wavelength for ugrizW1W2W3W4
    plt.figure(figsize=(16,9))
    plt.scatter(wavelengths_A, allfluxes, color='deeppink',marker = '*', s= 100, label='ubrite1')
    plt.plot(wavelengths_A[ind_sortedw], allfluxes[ind_sortedw], color='deeppink', label='ubrite1')
    plt.ylabel('Log Flux (nanomaggies)')
    plt.xlabel('Log Wavelength ($\AA$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # LB looking up the object in SDSS to get redshift and look at the spectrum
    print('The redshift of this object is ~1.035')
    print('Given how ubrite1 was selected, it makes sense that the spectrum looks this way, since it has notable continuum emission at shorter wavelengths, making the u higher.')




# In[ ]:




