import glob
import datetime as dt
import numpy as np
import os

def loaddriftruns(path='./figsfromcluster/figs/', name='time'):
  if name=='time':
    cd=dt.datetime.now()
    name=cd.strftime("Driftsimresults_%Y-%m-%d.npz")
  
  allnpz=glob.glob(os.path.join(path,'*R*.npz'))
  z=np.load(allnpz[0])
  finalpops=np.zeros([len(allnpz),10,10])
  envifull=np.zeros([len(allnpz), 100, 50])
  envimean=np.zeros([len(allnpz),50])
  envimeanvariance=np.zeros(len(allnpz))
  fbands=np.empty([len(allnpz)], dtype=str)
  fband=['0','1', '2', '3', '4', 'W', 'P']
  index=0
  errors=0
  print(len(allnpz))
  for f in fband:
    print(f)
    filenamepattern='F'+f+'_R*.npz'
    print(filenamepattern)
    x=glob.glob(os.path.join(path,filenamepattern))
    print(len(x))
    for i in range(len(x)):
        temp=np.load(x[i])
        envifull[i,:,:]=temp['envi']
        q=np.where(temp['envi']==1, temp['envi'], 0) #check here, may be a bug in what this does to envi
        r=np.argwhere(q[:,:]==1)
        if r.shape[0]==50:
          envimeanvariance[index]=temp['envimeanvariance']

          envimean[index,:]=r[np.argsort(r[:,1],0)][:,0]
      #     envimean[i,:]=np.max(temp['envi'], axis=0)
          finalpops[index,:,:]=temp['finalpopulations']
          fbands[index]=f
          # print(index)
          index+=1
        else:
          errors+=1
  print(errors)
  np.savez(name, 
  fbands=fbands[0:-errors], finalpopulations=finalpops[0:-errors],
   envimeanvariance=envimeanvariance[0:-errors], envifull=envifull[0:-errors,:,:], envimean=envimean[0:-errors],
    y_prefvariancemesh=z['prefvariancemesh'], x_driftvariancemesh=z['driftvariancemesh'])