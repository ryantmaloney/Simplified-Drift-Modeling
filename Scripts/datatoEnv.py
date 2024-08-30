# The goal here is to come up with some code to take an arbitrary time series and convert it to an env file usable with the simulation (simpledrift).

# Step 1 will be to write a function for an arbitrary (set?) of data. The data will be z-scored (so mean of the data=mean value). SD should *either* be based on individual data or maybe scaled across all data in that type (so that temperature depends on fluctuations. Maybe try both)

# Step 2 is determining the length of the data. 

# Step 3 is determining 

import string
# from flask import request_tearing_down
import numpy as np
import scipy.stats as sci
# import scipy.interpolate as ntr
import pandas as pd
import glob
import matrixmaker as mm
import xarray as xr

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

def folderToEnvs(folderpath):
  list=glob.glob(folderpath+"/*",  )

  #Here goes the for loop later, but for now, let's just focus on the first one
  dataTypeIndex=0
  longform=pathToPandas(list[dataTypeIndex], siteID="all")
  wf=longform.iloc[:,1:].pivot_table(index=["siteID", "variable"], columns=["date"])
  siteIndex=1
  rawenv=makeContinuous(wf.iloc[siteIndex,:])
  return rawenv

def folderToxArray(folderpath, chunklength=1000, datatypecap=10000, stationnumbercap=10000, chunkcap=100):
  masterlist=getEnvsFromFolder(folderpath)
  # firstentry=True
  for l, list in enumerate(masterlist):
    firstentry=True

    print(list+" (num "+str(l)+"/"+str(len(masterlist))+")")
    if l>datatypecap:
      break
    csv0=pathToPandas(list, siteID='all')
    uniquevalues=np.unique(csv0["siteID"])
    # print(uniquevalues)
    # print("Num unique sites "+str(len(uniquevalues)))
    for ui, u in enumerate(uniquevalues):
        if ui>stationnumbercap:
          break
        print(str(u)+" (num "+str(ui)+"/"+str(uniquevalues.shape[0])+")")
        numu=(np.count_nonzero(csv0["siteID"]==u))
        thischunk=csv0[(csv0["siteID"]==u)]
        numureal=np.count_nonzero(pd.notna(thischunk["value"]))
        # print(numureal)
        # print("Num real "+str(numureal)+"/"+str(numu))
        thischunk=csv0[(csv0["siteID"]==u)].interpolate(limit=5)
        # thischunk["value"].interpolate()
        #Calculate num valid starts
        valid_starts=np.zeros(len(thischunk))
        chunkstarts=np.zeros(len(thischunk))
        chunkcheck=0
        for i in np.arange(len(thischunk)-chunklength):

            if np.all(np.isfinite(thischunk['value'][i:i+chunklength])):
                valid_starts[i]=1
                if chunkcheck<=0:
                    chunkstarts[i]=1
                    chunkcheck=chunklength
            chunkcheck=chunkcheck-1

                # print("valid start at "+str(i))
        # plt.plot(valid_starts)
        # print(str(np.sum(valid_starts))+" valid starts")
        # print(str(np.sum(chunkstarts))+" chunks")l

        numchunks=np.floor(numu/chunklength)
        # thischunk=csv0[(csv0["siteID"]==u)]
        # print(thischunk)
        # for n in np.arange(numchunks):
        for n, startindex in enumerate(np.where(chunkstarts==1)[0]):
            # print(n)
            chunknum=np.ones(chunklength, dtype=int)*n
            # print(chunknum)
            chunkday=np.arange(chunklength)
            # print(n)
            # print(startindex)
            newchunk=thischunk[int(startindex):int(startindex+chunklength)].assign(chunk=chunknum).assign(day=chunkday)
            newchunk["value"]=newchunk["value"]-np.mean(newchunk["value"])
            newchunk["value"]=newchunk["value"]/np.std(newchunk["value"])

            # print(newchunk.columns)
            if newchunk.columns[2]=='date':
              newchunk["startDateTime"]=pd.to_datetime(newchunk["date"])
              newchunk.drop(labels='date', axis=1)
            else:
              newchunk["startDateTime"]=pd.to_datetime(newchunk["startDateTime"])
            newchunk["startDateTime"]
            newchunk.reset_index()
            # print(newchunk["startDateTime"][0])
            # newchunk["startDateTime"][0]
            # print(np.mean( newchunk["value"]))
            if firstentry:
                allfiles=newchunk
                firstentry=False
            else:

                # print(newchunk)
                allfiles=allfiles.append(newchunk)

        
        #Maybe check for gaps
    allfiles=allfiles.reset_index().loc[:,"siteID":]
    allfiles# print(numchunks)
    allfiles_trunc=allfiles.loc[:,{"siteID", "variable", "chunk", "day", "value"}]
    allfiles_nodups=allfiles_trunc.drop_duplicates()
    allfiles_nodups.to_csv("allfiles_1000days_"+str(l)+".csv")
  return allfiles_nodups


def makeMasterChunkList(folderpath):
  masterlist=glob.glob(folderpath+"/*",  )
  for l, list in enumerate(masterlist):
    print(l)
    longform=pd.read_csv(list).drop("Unnamed: 0", axis=1).pivot(index=["variable", "chunk", "siteID"], columns="day")
    if l==0:
      allfiles=longform
    else:
      allfiles=pd.concat([allfiles, longform], axis=0)
  return allfiles

def runSimulationOnline(csvfile, filesavedir, linenumber):
  excerpt=pd.read_csv(csvfile, skiprows=linenumber+2, nrows=1)
  variablename=excerpt.iloc[0,0]
  # print(variablename)
  chunk=excerpt.iloc[0,1]
  # print(chunk)
  siteID=excerpt.iloc[0,2]
  # print(siteID)
  envi=excerpt.iloc[0,3:]
  # name=excerpt[0, 0:3]
  # print(name.shape)
  name=filesavedir+"/"+str(variablename)+"_"+str(siteID)+"_c"+str(chunk)
  print(name)
  df=mm.matrixmaker(np.array(envi), bhstrats=np.linspace(0, .5, 11),
   savealldays=False, savedata=False, saveenv=False,
   driftmaxdistribution=.3, driftstrats=np.linspace(0, .1, 11), birthrate=[40],
   matureage=[10, 60, 360], envivariance=[.3], envimeanvariance=np.linspace(.1, .3, 3), nameprefix="test")
  day=np.arange(envi.shape[0])
  df=df.assign_coords(coords={"variable":variablename, "chunk":chunk, "siteID":siteID})
# print(days)
  envi2=xr.DataArray(envi,coords={"day":day})
  envi2=envi2.assign_coords(coords={"variable":variablename, "chunk":chunk, "siteID":siteID})
  fullDataset=xr.Dataset({"df":df.squeeze(),"envi":envi2})
  fullDataset.to_netcdf(name+".nc")
  # return fullDataset

def calcmaxes(pathtosimresults):
  a=xr.load_dataset(pathtosimresults)
  return a
  flag=False
  for emv in a["envmeanvar"]:
      for ma in a["matureage"]:
          singlerun=a.sel({"envmeanvar":emv}).sel({"matureage":ma})
          da=singlerun["df"].squeeze()
          # print(da)
          max1=da.max()

          emvi=np.expand_dims(emv.data, axis=0)
          mai=np.expand_dims(int(ma.data), axis=0)

          maxdrifti=np.expand_dims(np.array(singlerun.where(singlerun==max1, drop=True)['drift']), axis=1)
          maxbhi=np.expand_dims(np.array(singlerun.where(singlerun==max1, drop=True)['bet-hedging']), axis=1)
          # print(maxdrifti.shape)
          max_drift_i=xr.DataArray(data=maxdrifti,
          coords={"envmeanvar":emvi, "matureage":mai},
          name="maxdrifti")
          max_bh_i=xr.DataArray(data=maxbhi,
          coords={"envmeanvar":emvi, "matureage":mai},
          name="maxbhi")
          comb=xr.merge([max_drift_i, max_bh_i])
          if flag:
              # print(z)
              # print(max_drift_i)
              z=xr.merge([z, comb], join="outer")
          else:
              flag=True
              z=comb
          # b["max_drift_i"]=max_drift_i
          # b[{"maxpop_drift_index":max_drift_i}]

  result=a.merge(z)
  return result

def combinealldata(listoffiles):
  first=True
  for file in listoffiles:
    print(file)
    if first:
      a=calcmaxes(file).expand_dims(["variable", "chunk", "siteID"])
      a=a.stack(simkey=["variable", "chunk", "siteID"])
      first=False
    else:
      try:
        b=calcmaxes(file).expand_dims(["variable", "chunk", "siteID"])
      except:
        print("Problem with file %f", file)
      else:
        b=b.stack(simkey=["variable", "chunk", "siteID"])
      # print(b)
        a=a.merge(b, join="outer")

  return a

def calculatedriftbhmax(dataset):
  ## calculate max drift and bet-hedging for each envmeanvar and matureage
  b=dataset
  flag=False
  for emv in dataset["envmeanvar"]:
      for ma in dataset["matureage"]:
          singlerun=dataset.sel({"envmeanvar":emv}).sel({"matureage":ma})
          da=singlerun["df"].squeeze()
          # print(da)
          max1=da.max()

          emvi=np.expand_dims(emv.data, axis=0)
          mai=np.expand_dims(int(ma.data), axis=0)

          maxdrifti=np.expand_dims(np.array(singlerun.where(singlerun==max1, drop=True)['drift']), axis=1)
          maxbhi=np.expand_dims(np.array(singlerun.where(singlerun==max1, drop=True)['bet-hedging']), axis=1)
          # print(maxdrifti.shape)
          max_drift_i=xr.DataArray(data=maxdrifti,
          coords={"envmeanvar":emvi, "matureage":mai},
          name="maxdrifti")
          max_bh_i=xr.DataArray(data=maxbhi,
          coords={"envmeanvar":emvi, "matureage":mai},
          name="maxbhi")
          comb=xr.merge([max_drift_i, max_bh_i])
          if flag:
              # print(z)
              # print(max_drift_i)
              z=xr.merge([z, comb], join="outer")
          else:
              flag=True
              z=comb
          # b["max_drift_i"]=max_drift_i
          # b[{"maxpop_drift_index":max_drift_i}]

  b.merge(z)
  return b
  # max_drift_i
  # z