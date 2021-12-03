# The goal here is to come up with some code to take an arbitrary time series and convert it to an env file usable with the simulation (simpledrift).

# Step 1 will be to write a function for an arbitrary (set?) of data. The data will be z-scored (so mean of the data=mean value). SD should *either* be based on individual data or maybe scaled across all data in that type (so that temperature depends on fluctuations. Maybe try both)

# Step 2 is determining the length of the data. 

# Step 3 is determining 

import numpy as np
import scipy.stats as sci

def convertTimeSeriestoEnv(timeseries, scaling='single', length='all', numberofbins=100, maxsurvivalrate=1, envivariance=.1, envimeanvariance=1):
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

# def sinwaveinput(numberofbins, numberofdays, envimean, envivariance, maxsurvivalrate, gain, per):
#     envi=np.zeros((numberofbins,numberofdays))
#     x=np.linspace(-1,1,numberofbins)
#     envi[:,0]=sci.norm.pdf(x,envimean,envivariance) # A gaussian of environment with center around 0
#     envi=envi/(np.max(envi))*maxsurvivalrate # Normalizing the maximum envi value and factoring in deathrate
# 
#     for t in range(1,numberofdays):
#         envi[:,t]=sci.norm.pdf(x,(envimean+gain*np.sin(t*np.pi*2/per)),envivariance) # Making envi a sin wave that changes over time
#         envi[:,t]=envi[:,t]/np.max(envi[:,t])*maxsurvivalrate
  return(envi)