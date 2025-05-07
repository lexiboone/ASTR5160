#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.table import Table, vstack
import glob
from astropy.io import fits
from tasks.week12.flaggingdata import decode_sweep_name, is_in_box, findsweep
from homework.HW3Lexi import findsources, mag_nanomaggies
import astropy
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import os
import astropy.units as u
import numpy as np
from time import sleep 
import matplotlib.pyplot as plt
#from astropy.io import ascii
import argparse
import warnings


def crossmatchrevised(matching_radius, filenames, max_rband, objs_table):
    """Cross matches objects in the a quasar file with a sweep file.
    
    Parameters
    ----------
    matching_radius : :class:`float`
        Matching radius you want to use to match the objects in the surveys.
        All should be in arcseconds 
    filenames : :class:`list`
        A list of the Legacy Survey sweep files you want to match to the FIRST survey.
    max_rband: :class:`float`
        The maximum r and minimum W1-W2 colors you want to make your cuts around. If you want all objects with r mag < 22 your input should be 22, etc.
    objs_table : :class:'astropy table'
        objects in an atropy table that you want to find matches around
    

    Returns
    -------
    The matched objects from the sweep file, the matched objects in the original quasar file, and the objects which do not match from the sweep file. : :class:'Astropy Table'

    Notes
    ------
    We can assume everything else (the last table returned) are stars
    """
    
    # LB just changing naming
    firstsources_insurvey = objs_table

    # LB reading in the objects in the quasar file as a SkyCoord object
    RA_is = objs_table["RA"]
    Dec_is = objs_table["DEC"]
    firstsourcecoord = SkyCoord(RA_is, Dec_is, unit=(u.degree, u.degree))
    combinedis = np.column_stack((RA_is,Dec_is))

    # LB finding the sweep files we will need to match our objects
    sweepfiles_is = findsweep(firstsources_insurvey,filenames)

    # LB reading stacking all the necessary sweep files into one large file
    totfiles = vstack([Table.read(file, unit_parse_strict = 'silent', memmap=True) for file in sweepfiles_is])

    # LB making cuts based on flux and type to restrict the objects to the survey we are dealing with
    R_flux2 = totfiles["FLUX_R"]
    R_magnitudes2 =  mag_nanomaggies(R_flux2)
    R_index2 = (totfiles["TYPE"] == 'PSF') & (R_magnitudes2 < max_rband)
    totfiles = totfiles[R_index2]

    # LB making those objects Sky Coordinates
    totcoordinates = SkyCoord(totfiles["RA"], totfiles["DEC"], unit=(u.degree, u.degree))

    #totcoordinates = totcoordinates

    # LB finding the indeces of the matches in each file and indexing both tables
    id1, id2, d2, d3 = totcoordinates.search_around_sky(firstsourcecoord, u.arcsecond * matching_radius)
    matches = totfiles[id2]
    id3 = np.delete(np.arange(0,len(totfiles)), id2)
    stars = totfiles[id3]
    #print(len(matches))
    FIRSTmatches = firstsources_insurvey[id1]
    RA_match = matches["RA"]
    Dec_match = matches["DEC"]

    # LB uncomment for troubleshooting
    #print(len(stars))
    #print(len(matches))

    # LB return the sweepfile matches, the quasar file matches, and the sweep file non-matches
    return matches, FIRSTmatches, stars


def splendid_function(Table):
    """Classifies quasars in an astropy table
    
    Parameters
    ----------
    Table : :class:'astropy table'
        objects in an atropy table that you want classify as quasars or not
    

    Returns
    -------
    Boolean array of Trues and Falses. Trues correspond to quasars and Falses correspond to other objects in the file.

    Notes
    ------
    The input table must have the columns used to make color and flag cuts used below.
    """

    # LB uncomment to troubleshoot
    # colnames = Table.colnames
    # print('Colnames')
    # print(colnames)

    # LB defining the color and flag cuts we will use based on the columns in the sweeps files
    Objtype = Table["TYPE"]
    Gflux = Table['FLUX_G']
    Rflux = Table['FLUX_R']
    Zflux = Table['FLUX_Z']
    W1flux = Table['FLUX_W1']
    W2flux = Table['FLUX_W2']
    MWG = Table["MW_TRANSMISSION_G"]
    MWR = Table["MW_TRANSMISSION_R"]
    MWZ = Table["MW_TRANSMISSION_Z"]
    MWW1 = Table["MW_TRANSMISSION_W1"]
    MWW2 = Table["MW_TRANSMISSION_W2"]
    PMDEC = np.abs(Table['PMDEC'])
    PMRA = np.abs(Table['PMRA'])
    ALLMASKG = Table["ALLMASK_G"]
    ALLMASKR = Table["ALLMASK_R"]
    ALLMASKZ = Table["ALLMASK_Z"]
    WISE1MASK = Table["WISEMASK_W1"]
    WISE2MASK = Table["WISEMASK_W2"]
    maskbits = Table['MASKBITS']

    # LB converting to magnitude for the colors necessary
    corrected_W1_mag = mag_nanomaggies(W1flux/ MWW1)
    corrected_W2_mag = mag_nanomaggies(W2flux/ MWW2)
    corrected_G_mag = mag_nanomaggies(Gflux/ MWG)
    corrected_R_mag = mag_nanomaggies(Rflux/ MWR)
    corrected_Z_mag = mag_nanomaggies(Zflux/ MWZ)
    rmag = mag_nanomaggies(Rflux)
    

    # corrected_W3_mag = mag_nanomaggies(W3magcut/ MWW3)
    # corrected_W4_mag = mag_nanomaggies(W4magcut/ MWW4)

    # LB defining the parameters of the line we want to use to make our color cut
    bfunc = -1.0
    mfunc = 0.95
    mo = -2.5
    bo = -0.5
    #xfunc = np.linspace(-7, 7, len(Table['FLUX_R']))
    #dividefunc = mfunc * xfunc + bfunc

    # LB making the colors we want for our color cuts
    W1_W2 = corrected_W1_mag-corrected_W2_mag
    r_W1 = corrected_R_mag - corrected_W1_mag
    g_z = corrected_G_mag - corrected_Z_mag

    # LB indexing all the color and flag cuts to find our quasars
    boolindex = (Objtype == 'PSF') & (WISE2MASK == 0) & (WISE1MASK == 0) & (rmag < 19) &(PMDEC < 7) & (PMRA < 3) & (ALLMASKG == 0 ) & (ALLMASKR == 0 ) & (ALLMASKZ == 0) & (Gflux > 0) & (Rflux > 0 ) & (Zflux > 0) & (r_W1 > mfunc * g_z + bfunc) & (r_W1 > mo * g_z + bo) & (W1_W2 > -0.5) & (W1_W2 < 1.2) 


    # LB uncomment for troubleshooting
    #print(f'Table length with first bool is {len(Table[boolindex])}')
    #print(f' Total table length is {len(Table)}')

    # LB returning the boolean array of trues and falses
    return boolindex
    


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    # LB adding argument of a file path to argparse so when read a filepath at the CL it can classify quasars
    parser.add_argument("filepath", type=str, help="What is the full path to the sweeps file you want to classify quasars in?")
    args = parser.parse_args()
    FILEPATH = args.filepath
        
    # LB reading in the file as a table and passing it through splendid function
    objs_table = Table.read(FILEPATH, unit_parse_strict='silent', memmap=True)
    tabletf = splendid_function(objs_table)
    # LB printing the amount of quasars the function found in that sweep file.
    print(f'The total number of objects classified as quasars is {len(objs_table[tabletf])}')




    # COMMENT ALL THIS OUT SCRATCH WORK FOR MAKING CUTS
    # #determine_cuts('/d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits')
    # testfile = '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-180p020-190p025.fits'
    # objs_table = Table.read(testfile, unit_parse_strict='silent', memmap=True)
    # # print('SPLENDID FUNCTION')
    # # data = testing()
    # # print('TEST FUNCTION')
    # # splendid_results = splendid_function(data)
    # # count = splendid_results.sum()

    # #print(f"The number of objects classified as quasars is {count}")


    # tabletf = splendid_function(objs_table)
    # print(tabletf)
    # sweepfilenames = glob.glob('/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/*.fits')
    # filenameqsos = '/d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits'
    # qsostable = Table.read(filenameqsos, unit_parse_strict='silent')
    # # print('TESTING WITH RANDOM SWEEPS')
    # # splendid_function(objs_table)



    # #In[ ]:


    # crossmatchrevised(0.5, sweepfilenames, 19, qsostable)
    # sweepfilematches,qsosfile, stars = crossmatchrevised(0.5, sweepfilenames, 19, qsostable)
    # Gmagcut = sweepfilematches['FLUX_G']
    # Rmagcut = sweepfilematches['FLUX_R']
    # Zmagcut = sweepfilematches['FLUX_Z']
    # W1magcut = sweepfilematches['FLUX_W1']
    # W2magcut = sweepfilematches['FLUX_W2']
    # W3magcut = sweepfilematches['FLUX_W3']
    # W4magcut = sweepfilematches['FLUX_W4']

    # MWG = sweepfilematches["MW_TRANSMISSION_G"]
    # MWR = sweepfilematches["MW_TRANSMISSION_R"]
    # MWZ = sweepfilematches["MW_TRANSMISSION_Z"]
    # MWW1 = sweepfilematches["MW_TRANSMISSION_W1"]
    # MWW2 = sweepfilematches["MW_TRANSMISSION_W2"]
    # MWW3 = sweepfilematches["MW_TRANSMISSION_W3"]
    # MWW4 = sweepfilematches["MW_TRANSMISSION_W4"]

    # ALLMASKG = sweepfilematches["ALLMASK_G"]
    # ALLMASKR = sweepfilematches["ALLMASK_R"]
    # ALLMASKZ = sweepfilematches["ALLMASK_Z"]

    # PMDEC = sweepfilematches["PMDEC"]
    # print(f'MAX PM is {max(PMDEC)}')
    # print(f'MAX PM is {min(PMDEC)}')

    # allmask_options = np.arange(0,16)
    # def allmask_find(table, band):
    #     allmask_options = np.arange(0,16)
    #     for i in allmask_options:
    #         index = (table[band] == i)
    #         print(f'Table length for {i} in the {band} is {len(table[index])}')

    # allmask_find(sweepfilematches, "ALLMASK_G")

    # starGmagcut = stars['FLUX_G']
    # starRmagcut = stars['FLUX_R']
    # starZmagcut = stars['FLUX_Z']
    # starW1magcut = stars['FLUX_W1']
    # starW2magcut = stars['FLUX_W2']
    # starW3magcut = stars['FLUX_W3']
    # starW4magcut = stars['FLUX_W4']

    # starMWG = stars["MW_TRANSMISSION_G"]
    # starMWR = stars["MW_TRANSMISSION_R"]
    # starMWZ = stars["MW_TRANSMISSION_Z"]
    # starMWW1 = stars["MW_TRANSMISSION_W1"]
    # starMWW2 = stars["MW_TRANSMISSION_W2"]
    # starMWW3 = stars["MW_TRANSMISSION_W3"]
    # starMWW4 = stars["MW_TRANSMISSION_W4"]

    # starALLMASKG = stars["ALLMASK_G"]
    # starALLMASKR = stars["ALLMASK_R"]
    # starALLMASKZ = stars["ALLMASK_Z"]

    # corrected_g_mag = mag_nanomaggies(Gmagcut / MWG)
    # corrected_r_mag = mag_nanomaggies(Rmagcut / MWR)
    # corrected_z_mag = mag_nanomaggies(Zmagcut / MWZ)
    # corrected_W1_mag = mag_nanomaggies(W1magcut/ MWW1)
    # corrected_W2_mag = mag_nanomaggies(W2magcut/ MWW2)
    # corrected_W3_mag = mag_nanomaggies(W3magcut/ MWW3)
    # corrected_W4_mag = mag_nanomaggies(W4magcut/ MWW4)

    # starcorrected_g_mag = mag_nanomaggies(starGmagcut / starMWG)
    # starcorrected_r_mag = mag_nanomaggies(starRmagcut / starMWR)
    # starcorrected_z_mag = mag_nanomaggies(starZmagcut / starMWZ)
    # starcorrected_W1_mag = mag_nanomaggies(starW1magcut/ starMWW1)
    # starcorrected_W2_mag = mag_nanomaggies(starW2magcut/ starMWW2)
    # starcorrected_W3_mag = mag_nanomaggies(starW3magcut/ starMWW3)
    # starcorrected_W4_mag = mag_nanomaggies(starW4magcut/ starMWW4)

    # qsos_r_W1 = corrected_r_mag - corrected_W1_mag
    # qsos_g_z = corrected_g_mag - corrected_z_mag


    # star_r_W1 = starcorrected_r_mag - starcorrected_W1_mag
    # star_g_z = starcorrected_g_mag - starcorrected_z_mag

    # qsos_g_r = corrected_g_mag - corrected_r_mag
    # qsos_W1_W2 = corrected_W1_mag - corrected_W2_mag


    # star_g_r = starcorrected_g_mag - starcorrected_r_mag
    # star_W1_W2 = starcorrected_W1_mag - starcorrected_W2_mag

    # b = -1.0
    # m = 0.97
    # bt = 2.5 
    # mt = 1
    # mo = -2.5
    # bo = -0.5
    # x = np.linspace(-7, 7, 10000)
    # y = m * x + b
    # yt = mt * x + bt
    # yo = mo *x + bo
    # plt.figure(figsize=(16,9))
    # plt.scatter(star_g_z, star_r_W1, s=0.1, marker='*', c='darkviolet', label='Stars')
    # plt.scatter(qsos_g_z, qsos_r_W1, marker='d', c='dodgerblue', label='Quasars')
    # plt.plot(x,y, color='purple')
    # plt.plot(x,yt)
    # plt.plot(x, yo)
    # plt.xlabel('g - z')
    # plt.ylabel('r - W1')
    # plt.legend()
    # plt.savefig('gzrw1id3.png')
    # plt.show()

    # plt.figure(figsize=(16,9))
    # plt.scatter(star_g_r, star_W1_W2, s=0.1, marker='*', c='darkviolet', label='Stars')
    # plt.scatter(qsos_g_r, qsos_W1_W2, marker='d', c='dodgerblue', label='Quasars')
    # plt.axhline(y=-0.365, color='r', linestyle='--', label=f'y = -0.365')
    # plt.axhline(y=-0.365, color='r', linestyle='--', label=f'y = -0.365')
    # plt.axhline(y=max(qsos_W1_W2)+0.01, color='r', linestyle='--', label=f'y = {max(qsos_W1_W2)+0.01}')

    # #plt.plot(x,y)
    # plt.xlabel('g - r')
    # plt.ylabel('W1 - W2')
    # plt.legend()
    # plt.savefig('grw1W2id3.png')
    # plt.show()


    # foundqsos = objs_table[tabletf]
    # fqG = foundqsos['FLUX_G']
    # fqR = foundqsos['FLUX_R']
    # fqZ = foundqsos['FLUX_Z']
    # fqW1 = foundqsos['FLUX_W1']
    # fqW2 = foundqsos['FLUX_W2']

    # fqMWG = foundqsos["MW_TRANSMISSION_G"]
    # fqMWR = foundqsos["MW_TRANSMISSION_R"]
    # fqMWZ = foundqsos["MW_TRANSMISSION_Z"]
    # fqMWW1 = foundqsos["MW_TRANSMISSION_W1"]
    # fqMWW2 = foundqsos["MW_TRANSMISSION_W2"]

    # fq_g_mag = mag_nanomaggies(fqG / fqMWG)
    # fq_r_mag = mag_nanomaggies(fqR / fqMWR)
    # fq_z_mag = mag_nanomaggies(fqZ / fqMWZ)
    # fq_W1_mag = mag_nanomaggies(fqW1/ fqMWW1)
    # fq_W2_mag = mag_nanomaggies(fqW2 / fqMWW2)


    # fq_g_r = fq_g_mag - fq_r_mag
    # fq_W1_W2 = fq_W1_mag - fq_W2_mag

    # fq_r_W1 = fq_r_mag - fq_W1_mag
    # fq_g_z = fq_g_mag - fq_z_mag


    # plt.figure(figsize=(16,9))
    # plt.scatter(star_g_z, star_r_W1, s=0.1, marker='*', c='darkviolet', label='Stars')
    # plt.scatter(qsos_g_z, qsos_r_W1, marker='d', c='dodgerblue', label='Quasars')
    # plt.scatter(fq_g_z, fq_r_W1, marker='d', c='red', label='Found Quasars')

    # plt.plot(x,y, color='purple')
    # #plt.plot(x,yt)
    # plt.xlabel('g - z')
    # plt.ylabel('r - W1')
    # plt.legend()
    # plt.savefig('gzrw1id3TEST.png')
    # plt.show()


    # starW1W2i = np.argsort(star_W1_W2)
    # sorted_starW1W2 = star_W1_W2[starW1W2i]

    # qsosW1W2i = np.argsort(qsos_W1_W2)
    # sorted_qsosW1W2 = qsos_W1_W2[qsosW1W2i]

    # print(f'W1-W2 cuts : min {-0.365} max {max(qsos_W1_W2)+0.01}')

    # print(f'W1-W2 for star min={min(star_W1_W2)} maxish={sorted_starW1W2[-2]}')


    # print(f'W1-W2 for qsos min={min(qsos_W1_W2)} max={max(qsos_W1_W2)}')

    # colors = [Gmagcut,Rmagcut,Zmagcut,W1magcut,W2magcut,W3magcut,W4magcut]
    # colors_names = ['Gmagcut','Rmagcut','Zmagcut','W1magcut','W2magcut','W3magcut','W4magcut']
    # MWdust = [MWG, MWR, MWZ, MWW1, MWW2, MWW3, MWW4]
    # for i in range(len(colors)):
    #     corrected_color = mag_nanomaggies(colors[i]/MWdust[i])
    #     min_color = np.min(corrected_color)
    #     max_color = np.max(corrected_color)
    #     print(f'Min of {colors_names[i]} is {min_color}')
    #     print(f'Max of {colors_names[i]} is {max_color}')

    # print(min(Rmagcut))
    # print(max(Rmagcut))
    # #print(sweepfilematches.variables)
    # #In[ ]:




