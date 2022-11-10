
import sys
import os
sys.path.insert(0, '../Scripts/')

import simpledrift as sd
import matrixmaker as mm
import pandas as pd
import numpy as np
# This code is for running on the server a detailed 


i = os.getenv('SLURM_ARRAY_TASK_ID')
if ~isinstance(i, int):
  i=1
print(i)


# fbands=1/2**np.arange(1,11, 1)[:0:-1]
# per=1/fbands
# print(per)
# matureagematch=np.int64(per[-2:0:-1])


# Version for September 2022
fbands=1/2**np.arange(1,10.5, .5)[:0:-1]
per=1/fbands
print(per)
matureagematch=np.int64(per[-2:0:-1])
# print(matureagematch)
phasearray_size=fbands.shape[0]
envimeanvariance=np.linspace(0, .3, phasearray_size)
# #simple test
# fbands=1/2**np.arange(1,10, 2)[:0:-1]
# per=1/fbands
# print(per)
# matureagematch=np.int64(per[-2:0:-1])
# print(matureagematch)

mm.frequency_phaseplane(matureage=[10],
fbands=fbands,
numberofdays=101, bhmax=.3, driftmax=.3,
strategy_resolution=11,
numberofbins=200,
envivariance=[.2],
filename_prefix="FreqvsEnvironmentMean_101days_includingpref_.2ev_R"+str(i),
envimeanvariance=envimeanvariance,
savepath="../Results/10-20-22/",
savealldays=True,
saveallprefs=True,
birthrate=[20],
i=i
)