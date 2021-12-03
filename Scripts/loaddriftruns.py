import glob
import datetime as dt
import numpy as np
import os
import re
import string
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


def loadrealworlddriftruns(path='./figsfromcluster/figs/', name='time'):
  if name=='time':
    cd=dt.datetime.now()
    name=cd.strftime("Driftsimresults_%Y-%m-%d.npz")
  
  allnpz=glob.glob(os.path.join(path,'BR*MA*R*.npz'))
  print('Size of loaded files')
  print(len(allnpz))
  z=np.load(allnpz[0])
  bins=100
#   days=100
  days=len(z['envi'])

  finalpops=np.zeros([len(allnpz),10,10])
#   envifull=np.zeros([len(allnpz), bins, days])
  envifull=np.zeros([len(allnpz), days, bins])

  envimean=np.zeros([len(allnpz),days])
  envimeanvariance=np.zeros(len(allnpz))
  envivariance=np.zeros(len(allnpz))
  birthrate=np.zeros(len(allnpz))
  matureage=np.zeros(len(allnpz))
#   fbands=np.empty([len(allnpz)], dtype=str) #Only if W and P are options
  fbands=np.empty([len(allnpz)])

  
  #I should set this to automatically figure out based on what fs exist 
#   fband=['0','1', '2', '3', '4', 'W', 'P']
#   p = re.compile(r'F(?P<num>\d*)')
  p = re.compile(r'BR\d{1,3}MA\d{1,3}F(?P<num>[^_]*)')
  


  findexes=[]
  for l in allnpz:
    m=p.search(l)
    findexes.append(m.group('num'))
  fband=np.unique(findexes)
  print("Fbands:")
  print(fband)
  
  
  brband=['1', '10', '100']
  
  p = re.compile(r'BR\d{0-3}(?P<num>\d*)')
  findexes=[]
  for l in allnpz:
    m=p.search(l)
    findexes.append(m.group('num'))
  brband=np.unique(findexes)
  print("Brbands:")
  print(brband)
  
  p = re.compile(r'MA(?P<num>\d*)')
  findexes=[]
  for l in allnpz:
    m=p.search(l)
    findexes.append(m.group('num'))
  maband=np.unique(findexes)
  print("MABands")
  print(maband)
  
  index=0
  errors=0
  print(len(allnpz))
  for br in brband:
    for f in fband:
      # Here's where we load everything up  for f in fband:
      print(f)
      filenamepattern='BR'+br+'M*F'+f+'_R*.npz'
      print(filenamepattern)
      x=glob.glob(os.path.join(path,filenamepattern)) #Find all filename patterns for given frequency
      print(len(x))
      for i in range(len(x)):
  #       print(x[i])
        temp=0
        try: #This is checking for files that don't load
          temp=np.load(x[i], allow_pickle=True) #Something complaining about pickles here, only for some files
        except:
          print("Didn't load "+str(x[i]))
          errors+=1 
        else:
          envifull[i,:,:]=temp['envi']
          q=np.where(temp['envi']==1, temp['envi'], 0) #check here, may be a bug in what this does to envi
          r=np.argwhere(q[:,:]==1)
          if r.shape[0]==days:
            print('Loading '+x[i])
            envimeanvariance[index]=temp['envimeanvariance']
            matureage[index]=temp['matureage']
            birthrate[index]=temp['birthrate']
            envivariance=temp['envivariance']
            envimean[index,:]=r[np.argsort(r[:,1],0)][:,0]
        #     envimean[i,:]=np.max(temp['envi'], axis=0)
            finalpops[index,:,:]=temp['finalpopulations']
            print(index)
            print(f)
            
            fbands[index]=f
            index+=1
          else:
            errors+=1 

  print("Total of "+ str(index)+ " runs successfully loaded")
  print("Didn't load "+str(errors)+" runs")
  np.savez(name, 
  fbands=fbands[0:-errors], finalpopulations=finalpops[0:-errors],
    envimeanvariance=envimeanvariance[0:-errors], envifull=envifull[0:-errors,:,:], envimean=envimean[0:-errors], 
    matureage=matureage[0:-errors], birthrate=birthrate[0:-errors],
    y_prefvariancemesh=z['prefvariancemesh'], x_driftvariancemesh=z['driftvariancemesh'])
  print('saved at ' + name)

# For loading normal runs

def loaddriftruns(path='./figsfromcluster/figs/', name='time'):
  if name=='time':
    cd=dt.datetime.now()
    name=cd.strftime("Driftsimresults_%Y-%m-%d.npz")
  
  allnpz=glob.glob(os.path.join(path,'BR*MA*R*.npz'))
  z=np.load(allnpz[0])
  print(z['envi'].shape)
  days=(z['envi'].shape[1])
  print(days)
  bins=100
#   days=100
  finalpops=np.zeros([len(allnpz),10,10])
  envifull=np.zeros([len(allnpz), bins, days])
#   envifull=np.zeros([len(allnpz), days, bins])

  envimean=np.zeros([len(allnpz),days])
  envimeanvariance=np.zeros(len(allnpz))
  envivariance=np.zeros(len(allnpz))
  birthrate=np.zeros(len(allnpz))
  matureage=np.zeros(len(allnpz))
  freqmin=np.zeros(len(allnpz))
  freqmax=np.zeros(len(allnpz))
#   fbands=np.empty([len(allnpz)], dtype=str) #Only if W and P are options
  fbands=np.empty([len(allnpz)])
  rindex=np.empty([len(allnpz)])

  
  #I should set this to automatically figure out based on what fs exist 
#   fband=['0','1', '2', '3', '4', 'W', 'P']
  p = re.compile(r'BR\d{1,3}MA\d{1,3}F(?P<num>[^_]*)')
  findexes=[]
  for l in allnpz:
    m=p.search(l)
    findexes.append(m.group('num'))
  fband=np.unique(findexes)
  print("Fbands:")
  print(fband)
  
  
  brband=['1', '10', '100']
  
  p = re.compile(r'BR(?P<num>\d*)')
  findexes=[]
  for l in allnpz:
    m=p.search(l)
    findexes.append(m.group('num'))
  brband=np.unique(findexes)
  print("Brbands:")
  print(brband)
  
  p = re.compile(r'MA(?P<num>\d*)')
  findexes=[]
  for l in allnpz:
    m=p.search(l)
    findexes.append(m.group('num'))
  maband=np.unique(findexes)
  print("MABands")
  print(maband)
  
#   p = re.compile(r'R(?P<num>\d*)')
#   rindexes=[]
#   for l in allnpz:
#     m=p.search(l)
#     rindexes.append(m.group('num'))
#   maband=np.unique(findexes)
#   print("MABands")
#   print(maband)
  
  index=0
  errors=0
  print(len(allnpz))
  for br in brband:
    for f in fband:
      # Here's where we load everything up  for f in fband:
      print(f)
      filenamepattern='BR'+br+'M*F'+f+'_R*.npz'
      print(filenamepattern)
      x=glob.glob(os.path.join(path,filenamepattern)) #Find all filename patterns for given frequency
      print(len(x))
      for i in range(len(x)):
  #       print(x[i])
        temp=0
        try: #This is checking for files that don't load
          temp=np.load(x[i], allow_pickle=True) #Something complaining about pickles here, only for some files
        except:
          print("Didn't load "+str(x[i]))
          errors+=1 
        else:
          print(envifull.shape)
          print(temp['envi'].shape)
          envifull[i,:,:]=temp['envi']
          q=np.where(temp['envi']==1, temp['envi'], 0) #check here, may be a bug in what this does to envi
          r=np.argwhere(q[:,:]==1)
          if r.shape[0]==days:
            print('Loading '+x[i])
            envimeanvariance[index]=temp['envimeanvariance']
            matureage[index]=temp['matureage']
            birthrate[index]=temp['birthrate']
            freqmin[index]=temp['freqmin']
            freqmax[index]=temp['freqmax']
            envivariance=temp['envivariance']
            envimean[index,:]=r[np.argsort(r[:,1],0)][:,0]
        #     envimean[i,:]=np.max(temp['envi'], axis=0)
            finalpops[index,:,:]=temp['finalpopulations']
            print(index)
            print(f)
            ret=re.search('_R(?P<num>\d[^_]*)', str(x[i]))
            rindex[index]=ret.group('num')
            fbands[index]=f
            index+=1
          else:
            errors+=1 

  print("Total of "+ str(index)+ " runs successfully loaded")
  print("Didn't load "+str(errors)+" runs")
  np.savez(name, 
  fbands=fbands[0:index-errors], finalpopulations=finalpops[0:index-errors],
    envimeanvariance=envimeanvariance[0:index-errors], envifull=envifull[0:index-errors,:,:], envimean=envimean[0:index-errors], freqmin=freqmin[0:index-errors], freqmax=freqmax[0:index-errors],
    matureage=matureage[0:index-errors], birthrate=birthrate[0:index-errors],
    rindex=rindex[0:index-errors],
    y_prefvariancemesh=z['prefvariancemesh'], x_driftvariancemesh=z['driftvariancemesh'])
  print('saved at ' + name)
  return allnpz
  
def row_col_argmax(a):
    '''
    Given a 2-D array a,
    returns the row index + column index
    of the array's maximum
    from here: https://stackoverflow.com/a/9483964
    '''
    return np.unravel_index(a.argmax(), a.shape)

def using_multiindex(A, columns):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s)for s in shape], names=columns)
    df = pd.DataFrame({'FinalPops': A.flatten()}, index=index).reset_index()
    return df

def row_col_argmax(a):
    '''
    Given a 2-D array a,
    returns the row index + column index
    of the array's maximum
    from here: https://stackoverflow.com/a/9483964
    '''
    return np.unravel_index(a.argmax(), a.shape)
    
def npztotable(npzfile):
  x=np.load(npzfile)
  x_driftvariancemesh=x['x_driftvariancemesh']
  y_prefvariancemesh=x['y_prefvariancemesh']
  
  finalpopulations=x['finalpopulations']
  maxpop=np.array([row_col_argmax(fp) for fp in finalpopulations])
# CHECK THAT THESE ARE RIGHT VIGOUROUSLY
  max_bh=maxpop[:,1]
  max_drift=maxpop[:,0]
  
  d= {"fbands": x['fbands'],"freqmax": x['freqmax'], "freqmin":x['freqmin'],
    "envimeanvariance": x['envimeanvariance'],
#     'envifull':x['envifull'],
#     'envimean':x['envimean']
   "matureage":x['matureage'],
    "birthrate":x['birthrate'],
    "max_drift":max_drift,
    "max_bethedging":max_bh,
    "rindex":x['rindex']
   }
  drifttable=pd.DataFrame(d)
#   drifttable
  df=using_multiindex(finalpopulations, ['index', 'rows', 'columns'])
  drifttable2=drifttable
  drifttable2.reset_index(inplace=True)
  drft=pd.merge(drifttable2, df, on="index", how='outer')
  drftpiv=drft.pivot_table(values="FinalPops", index=["fbands", "birthrate", "rows"], columns=["columns"])

  fxma=drft.iloc[:,[1,3,7,8,9]]
  
# This right here is wrong
#   averagebh=drft.pivot_table(values="FinalPops", index=["index", "rows"])
#   averagedrift=drft.pivot_table(values="FinalPops", index=["index", "columns"])

  averagebh=drft.pivot_table(values="FinalPops", index=["index", "columns"])
  averagedrift=drft.pivot_table(values="FinalPops", index=["index", "rows"])

  averagepop=drft.pivot_table(values="FinalPops", index=["index"])
  averagedrift=averagedrift/averagepop
  averagebh=averagebh/averagepop

  # weightedaveragedrift=averagedrift*
  # foo=pd.DataFrame(averagedrift.index.get_level_values(1))
  driftind=np.array(averagedrift.index.get_level_values(1))
  bhind=np.array(averagebh.index.get_level_values(1))

  weightedaveragedrift=averagedrift['FinalPops']*driftind
  weightedaveragebh=averagebh['FinalPops']*bhind
  # weightedaveragedrift=averagedrift*foo
  weightedaveragebh
  weightedaveragebh=weightedaveragebh.mean(level=0)
  weightedaveragedrift=weightedaveragedrift.mean(level=0)
  drifttable['DriftWA']=weightedaveragedrift
  drifttable['BHWA']=weightedaveragebh
  #Gmeans are wrong: also
  drifttable['DriftWGM']=stats.gmean(averagedrift['FinalPops']*driftind)
  drifttable['BHWGM']=stats.gmean(averagebh['FinalPops']*bhind)
  drifttable['Median Frequency']=(drifttable['freqmax']+drifttable['freqmax'])/2
  drifttable['Median Period']=np.round(1/drifttable['Median Frequency'], 1)
#   drifttable
  return drifttable


def makephaseplane(drifttable, plotvariables=["Median Period", "matureage"]):
  phaseplane=drifttable.pivot_table(values=["DriftWA", "BHWA"], index=plotvariables)
  phaseplane.reset_index(inplace=True)

  #Here's where we need to start changing variables 
  phaseplanemax=drifttable.pivot_table(values=["max_drift", "max_bethedging"], index=plotvariables)
  phaseplanemax.reset_index(inplace=True)
  ppmmax=phaseplanemax.melt(id_vars=plotvariables)
#   print(ppmmax)
#   ppmmax['Median Frequency']=(31-ppmmax['fbands']*2)
  
#   phaseplanemax
  ppm=phaseplane.melt(id_vars=plotvariables)
  print(ppm)
  dmaxhm=ppmmax[ppmmax['variable']=="max_drift"].pivot(index=plotvariables[0], columns=plotvariables[1], values="value")
  bmaxhm=ppmmax[ppmmax['variable']=="max_bethedging"].pivot(index=plotvariables[0], columns=plotvariables[1], values="value")
  ppmfc=ppm
#   ppmfc['Median Frequency']=(31-ppm['fbands']*2)
  bhwahm=ppmfc[ppmfc['variable']=="BHWA"].pivot(index=plotvariables[0], columns=plotvariables[1], values="value")
  dwahm=ppmfc[ppmfc['variable']=="DriftWA"].pivot(index=plotvariables[0], columns=plotvariables[1], values="value")
#   print(dwahm)

  g=sns.heatmap(dwahm, cbar_kws={'label': 'Drift Index (0-10)'})
  g.set(xlabel=plotvariables[1], 
        ylabel=plotvariables[0],
        title='Weighted Average Drift Strategy')
        
  plt.show()
  
#   print(dwahm)
  g=sns.heatmap(bhwahm, cbar_kws={'label': 'BH Index (0-10)'})
  g.set(xlabel=plotvariables[1], 
      ylabel=plotvariables[0],
      title='Weighted Average Bet-Hedging Strategy')
      
  plt.show()
#   g=sns.heatmap(dmaxhm.iloc[:,0:-1], cbar_kws={'label': 'Mean Drift Peak'})
  g=sns.heatmap(dmaxhm, cbar_kws={'label': 'Mean Drift Peak'})
  g.set(xlabel=plotvariables[1], 
        ylabel=plotvariables[0],
        title='Mean Ideal Drift Strategy')
        
  plt.show()
#   g=sns.heatmap(bmaxhm.iloc[:,0:-1], 
  g=sns.heatmap(bmaxhm, cbar_kws={'label': 'Mean BH Peak'})
  g.set(xlabel=plotvariables[1], 
        ylabel=plotvariables[0],
        title='Mean Ideal Bet-Hedging Strategy')
  plt.show()
#   print(bmaxhm)
  return bmaxhm, dmaxhm, ppm.iloc[:,0], ppm.iloc[:,1]