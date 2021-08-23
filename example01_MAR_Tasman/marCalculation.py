#!/usr/bin/env python 
# version 3.6
"""
    Calculate and plot mass accumulation rates for Tasman Sea
    Sutherland et al. (2021) Paleoceanography Paleoclimatology
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
#plt.style.use('ggplot')
rc('font',size=10,family='Arial')
rc("pdf", fonttype=42)

import marFunctions as mf

__author__ = "Rupert Sutherland"

regions = ['LHRN','NCTN','LHRS','REIN']
xmax = 50
ymax = [100,120,80,300]
ageGrid = np.arange(0,50,0.1)
 
for i,region in enumerate(regions):
    
    print('Working on region',region)    
    # force unicode text, up to 99 characters for each site name
    # force it to be at least 1d, so iterable even if only one site
    siteList = np.atleast_1d(np.loadtxt('./siteList_'+region+'.txt',dtype='U99'))   
    interpFileName = './seismic_horizons_'+region+'.csv'    
    ageFileName = './seismic_ages_'+region+'.csv'

    # Setup plot for the region
    plt.figure(figsize=(6,4))
    plt.title(region)
    plt.xlim([0,xmax])
    plt.xticks(range(0,xmax+1,5))
    plt.ylim([0,ymax[i]])
    plt.yticks(range(0,ymax[i]+1,20))
    plt.minorticks_on()
    plt.xlabel('Age (Ma)')
    plt.ylabel ('Mass Accumulation Rate (kg $\mathregular{kyr^{-1}\ m^{-2}}$)')

    # boreholes
    marGridList = list()
    for site in siteList:
        depthAgeFileName = 'site_' + site + '_depth_age.txt'
        age,lsr,mar = mf.marFromDepthAge(depthAgeFileName)
        # make very small age step at unit boundaries, for interpolation process
        age = mf.fixMonotonic(age)
        interpFunc=interp1d(age,mar,bounds_error=False,fill_value=np.nan)
        marGrid = interpFunc(ageGrid)
        plt.plot(ageGrid,marGrid,label=site)
        mf.writeCSV('marSite_'+site+'.csv',[ageGrid,marGrid],['age','mar'])
        marGridList.append(marGrid)
        
    marSitesMean = np.nanmean(np.asarray(marGridList),axis=0)
    mf.writeCSV('marSitesMean_'+region+'.csv',[ageGrid,marSitesMean],['age','mar'])
      
    # seismic reflection units 
    ageMean,ageDiff,lsrMean,lsrStd,marMean,marStd,nPicks = \
                        mf.marFromHorizons(ageFileName,interpFileName)
    
    plt.errorbar(ageMean, marMean, xerr=ageDiff, yerr=marStd, 
                 label='Seismic\nN={:,}'.format(nPicks),
                 fmt='none',linestyle='none',color='darkgrey')
    mf.writeCSV('marSeismicHorizons_' + region + '.csv', 
             [ageMean, marMean, ageDiff, marStd],
             ['ageMean', 'marMean', 'ageDiff', 'marStd'])
    
    # interpolate each seismic horizon MAR onto regular age grid and average
    marGridList = list()
    for j in range(len(ageMean)):
        interpFunc=interp1d((ageMean[j]-ageDiff[j], ageMean[j]+ageDiff[j]), 
                            (marMean[j], marMean[j]),
                             bounds_error=False,fill_value=np.nan)
        marGridList.append(interpFunc(ageGrid))   
    marSeismicMean = np.nanmean(np.asarray(marGridList),axis=0)
    plt.plot(ageGrid,marSeismicMean,label='Mean Seismic',linewidth=3,color='black')
    mf.writeCSV('marSeismicMean_'+region+'.csv',
             [ageGrid,marSeismicMean],['age','mar'])
    
    # average boreholes and seismic
    marBest = np.nanmean(np.asarray([marSitesMean,marSeismicMean]),axis=0)
    plt.plot(ageGrid,marBest,label='Mean Value',linewidth=4,color='darkred')
    mf.writeCSV('marBest_'+region+'.csv',[ageGrid,marBest],['age','mar'])
    
    plt.legend()
    plt.savefig('marFig_'+region+'.jpg')
    plt.savefig('marFig_'+region+'.pdf')