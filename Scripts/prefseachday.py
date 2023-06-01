
import sys
import os
sys.path.insert(0, '../Scripts/')

import simpledrift as sd
import matrixmaker as mm
import pandas as pd
import numpy as np
from datetime import date
# This code is for running on the server a detailed 


i = os.getenv('SLURM_ARRAY_TASK_ID')
if isinstance(i, str):
  try:
    i=int(i)
  except:
    print("couldn't parse i, defaulting to 1")
    i=1

else:
  i=1

# fbands=1/2**np.arange(1,11, 1)[:0:-1]
# per=1/fbands
# print(per)
# matureagematch=np.int64(per[-2:0:-1])


# Version for September 2022
fbands=1/2**np.arange(5,10.5, .5)[:0:-1]
per=1/fbands
print(per)
matureagematch=np.int64(per[-2:0:-1])
# print(matureagematch)
phasearray_size=fbands.shape[0]
envimeanvariance=np.linspace(0, .3, phasearray_size)

savepath="../Results/10-22-22/"

savepath="../Results/"+str(date.today())+"/"


if os.path.exists(savepath):
  print("Saving at "+savepath)
else:
  print("Making "+savepath)
  os.mkdir(savepath)

for i in range(20):

  mm.frequency_phaseplane(matureage=[10],
  fbands=fbands,
  numberofdays=101, bhmax=.3, driftmax=.15,
  strategy_resolution=41,
  numberofbins=200,
  envivariance=[.2],
  filename_prefix="Freq.2emv_101days_includingpref_.15ev_R"+str(i),
  envimeanvariance=[.2],
  savepath=savepath,
  savealldays=True,
  saveallprefs=True,
  birthrate=[40],
  i=i
  )