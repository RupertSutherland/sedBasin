#!/usr/bin/env python 
# version 3.6
"""
    Calculate and plot mass accumulation rates for Tasman Sea
    Sutherland et al. (2021) Paleoceanography Paleoclimatology
"""

import pandas as pd
import numpy as np

import sedBasin as sed

__author__ = "Rupert Sutherland"

# Velocity model from IODP Site U1507 (deep hole through ooze-chalk)
# parameters (A,B,C): A*t**2 + B*t + C = z
# t is twt bsf (ms) and z is depth bsf (m)
poly_coeff = (306.6e-6,823.4e-3,0.0)
water_sound_velocity = 1500.

# Compaction parameters from IODP Site U1507 (deep hole through ooze-chalk)
compactionLength = 975.
surfacePorosity = 0.66
grainDensity = 2700.

def marFromHorizons(ageFileName,interpFileName):
    '''
    Calculates linear sedimentation rates and mass accumulation rates from
    seismic reflection interpretations tied to borehole ages.
    ARGS:
        ageFileName     CSV file with 'Name' and 'Age' columns
        interpFileName  CSV file with 'X','Y','Name1','Name2',... columns
        where twt values (ms) are in column 'Name1'
    RETURNS:
        ageMean,ageDiff,lsrMean,lsrStd, marMean,marStd  
    '''
    # Load data
    horizon = pd.read_csv(ageFileName,comment='#')
    twt = pd.read_csv(interpFileName,comment='#')
    
    # n is number of horizons, including seabed
    n = len(horizon)
    
    # first horizon is the seabed - read and store
    age1 = horizon['Age'][0]
    horizon1 = horizon['Name'][0]
    twtSeabed = np.array(twt[horizon1], dtype=float)
    twtBsf1 = twtSeabed * 0.
    depthBsf1 = twtBsf1
    
    ageMean = list()
    ageDiff = list()
    lsrMean = list()
    lsrStd  = list()
    marMean = list()
    marStd  = list()
    
    # loop through each horizon and analyze it, saving to compare with next
    nPicks=0
    for i in range(1,n):
        try:
            horizon2 = horizon['Name'][i].strip()
            age2 = horizon['Age'][i]
            
            # Ignore inversions in age
            if age2 <= age1 :
                age1 = age2
                horizon1 = horizon2
                # Reload twt below seafloor (bsf)
                twtBsf1 = np.array(twt[horizon1], dtype=float) - twtSeabed
                # Convert twt to depth bsf using a velocity model
                depthBsf1 = sed.depthBsf_from_twtBsf(twtBsf1,poly_coeff)             
                continue
            
            print( horizon1 + ' to ' + horizon2 )
            
            # Convert twt to below seafloor (bsf)
            twtBsf2 = np.array(twt[horizon2], dtype=float) - twtSeabed
            # Convert twt to depth bsf using a velocity model
            depthBsf2 = sed.depthBsf_from_twtBsf(twtBsf2,poly_coeff)  
            
            thickness = depthBsf2 - depthBsf1
            grainheight = sed.grainHeight(depthBsf1,thickness,
                                            surfacePorosity,compactionLength)    
            
            ageMean.append(0.5*(age2 + age1))
            ageDiff.append(0.5*(age2 - age1))
            
            # linear sedimentation rate (lsr)
            lsr = (thickness/(age2-age1))
            lsrMean.append(np.nanmean(lsr))    
            lsrStd.append(np.nanstd(lsr))
            
            # mass accumulation rate (mar)
            mar = 0.001 * grainDensity * grainheight/(age2-age1)
            marMean.append(np.nanmean(mar))    
            marStd.append(np.nanstd(mar))
            print(ageMean[-1], age1, age2, ' MAR=', np.nanmean(mar), 
                  ' n=',np.count_nonzero(~np.isnan(mar)),'\n')
            
            nPicks = nPicks + np.count_nonzero(~np.isnan(mar))
            # Remember bottom horizon becomes the top one in next loop
            age1 = age2
            horizon1 = horizon2
            depthBsf1 = depthBsf2
            
        except:
            print('PROBLEM WITH HORIZON: ' + horizon['Name'][i])
        
    print('nPicks =',nPicks)
    return ageMean,ageDiff,lsrMean,lsrStd,marMean,marStd,nPicks

def marFromDepthAge(depthAgeFileName):
    '''
    Calculates linear sedimentation rates (lsr) and mass accumulation rates 
    (mar) from a borehole depth,age table (file).
    ARGS:
        depthAgeFileName     tab delimited text with 'm_bsf','Ma'
    RETURNS:
        age,lsr,mar  
    Lines that start with '#' are treated as comments
    '''
    df = pd.read_table(depthAgeFileName,sep='\t',comment='#')
    
    age = list()
    lsr = list()
    mar = list()
    
    for i in range(1,len(df)):
        
        thickness = df['m_bsf'][i] - df['m_bsf'][i-1]
        ageDifference = df['Ma'][i] - df['Ma'][i-1]
        
        grainheight = sed.grainHeight(df['m_bsf'][i-1],thickness,
                                        surfacePorosity,compactionLength)    
        lsrVal = thickness/ageDifference
        marVal = 0.001 * grainDensity * grainheight/ageDifference
        
        age.extend((df['Ma'][i-1],df['Ma'][i]))
        lsr.extend((lsrVal,lsrVal))
        mar.extend((marVal,marVal))
        
    return age,lsr,mar

def fixMonotonic(array,dx=1e-6):
    '''
    Assumes array is already monotonic, but may have repeating equal values.
    In this case it will add dx to the second value, so no value is equal.
    ARGS:
        array assumed to be 1d.
    RETURNS:
        newarray which has no equal values in it.
    '''
    for i in range(1,len(array)):
        if array[i] == array[i-1]:
            array[i] = array[i-1] + dx
    return array

def writeCSVmar(FileName,ageGrid,marGrid):
    '''
    Creates a csv file from the age,mar arrays or lists
    '''
    df = pd.DataFrame(np.column_stack((ageGrid,marGrid)),columns=['age','mar'])
    df.to_csv(FileName)

def writeCSV(FileName,arrayList,columnNameList):
    '''
    Creates a csv file from list of array-like objects and list of column names
    ARGS:
        FileName string
        arrayList list of array-like objects
        columnNameList list of strings i.e. names of each array-like object
    RETURNS:
        Pandas DataFrame object
    '''
    df = pd.DataFrame(np.column_stack(arrayList),columns=columnNameList)
    df.to_csv(FileName)
    return df
