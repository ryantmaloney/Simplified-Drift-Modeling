# The goal here is to come up with some code to take an arbitrary time series and convert it to an env file usable with the simulation (simpledrift).

# Step 1 will be to write a function for an arbitrary (set?) of data. The data will be z-scored (so mean of the data=mean value). SD should *either* be based on individual data or maybe scaled across all data in that type (so that temperature depends on fluctuations. Maybe try both)

# Step 2 is determining the length of the data. 

# Step 3 is determining 

import string
import numpy as np
import scipy.stats as sci
# import scipy.interpolate as ntr
import pandas as pd
import glob

def pathToPandas(path, variablename=0, siteID=0):
  csv=pd.read_csv(path)
  uniquesites=np.unique(csv['siteID'])
  uniquevariables=np.unique(csv['variable'])
  if variablename=='all':
    csv=csv
  elif isinstance(variablename, int): 
    csv=csv[csv['variable']==uniquevariables[0]]
  elif isinstance(variablename, str):
    csv=csv[csv['variable']==variablename]
  
  if siteID=='all':
    csv=csv
  elif isinstance(siteID, int): 
    csv=csv[csv['siteID']==uniquesites[0]]
  elif isinstance(variablename, str):
    csv=csv[csv['siteID']==siteID]
  return csv

def convertTimeSeriestoEnv(timeseries, scaling='single', length='all', minrunlength=365, numberofbins=100, maxsurvivalrate=1, envivariance=.1, envimeanvariance=.1):
  # print(timeseries)
  timeseries=makeContinuous(timeseries, minrunlength=minrunlength)
  timeseries=timeseries-np.mean(timeseries)

  if scaling=='single':
    scaling=np.std(timeseries)
  timeseries=timeseries/scaling*envimeanvariance
  
  if length=='all':
    length=timeseries.shape[0]
  elif length>timeseries.shape[0]:
    length=timeseries.shape[0]
    print('Shortened length to '+str(length))
  actualstart=0
  if length<timeseries.shape[0]:
    possiblestarts=timeseries.shape[0]-length
    actualstart=np.random.randint(0, possiblestarts)
  #now convert time series to env
  envi=np.zeros((numberofbins, length))
  x=np.linspace(-1,1,numberofbins)
  envi[:,0]=sci.norm.pdf(x,timeseries[0],envivariance) # A gaussian of environment with center around 0
  envi=envi/(np.max(envi))*maxsurvivalrate # Normalizing the maximum envi value and factoring in deathrate
  for t in range(1, length):
    envi[:,t]=sci.norm.pdf(x,timeseries[t+actualstart], envivariance)
    envi[:,t]=envi[:,t]/np.max(envi[:,t])*maxsurvivalrate

  #switch this to just timeseries
  envi=timeseries

# def sinwaveinput(numberofbins, numberofdays, envimean, envivariance, maxsurvivalrate, gain, per):
#     envi=np.zeros((numberofbins,numberofdays))
#     x=np.linspace(-1,1,numberofbins)
#     envi[:,0]=sci.norm.pdf(x,envimean,envivariance) # A gaussian of environment with center around 0
#     envi=envi/(np.max(envi))*maxsurvivalrate # Normalizing the maximum envi value and factoring in deathrate
# 
#     for t in range(1,numberofdays):
#         envi[:,t]=sci.norm.pdf(x,(envimean+gain*np.sin(t*np.pi*2/per)),envivariance) # Making envi a sin wave that changes over time
#         envi[:,t]=envi[:,t]/np.max(envi[:,t])*maxsurvivalrate
  return(timeseries)

def makeContinuous(timeseries, interplength=5, minrunlength=365):
  startlocations=np.zeros(timeseries.shape[0]-minrunlength)
  #Find contin
  timeseriespd=pd.DataFrame(timeseries)
  timeseriespd=timeseriespd.interpolate(limit=interplength)

  for m in np.arange(timeseriespd.shape[0]-minrunlength):
    # print(m)
    startlocations[m]=np.all(np.isfinite(timeseriespd[m:m+minrunlength]))

  # for i in np.isfinite(timeseries):
  #   timeseries[i]=
  # print(np.nonzero(startlocations))
  startlocationindexes=np.nonzero(startlocations)[0]
  # start=np.amin(startlocationindexes)
  
  start=startlocationindexes[np.random.randint(0, startlocationindexes.shape[0])]

  samplearray=timeseriespd[start:start+minrunlength]

  return np.array(samplearray)

def getEnvsFromFolder(folderpath):
  list=glob.glob(folderpath+"/*",  )
  return list

# def csvToInput