
import sys
import os
sys.path.insert(0, '../Scripts/')

import simpledrift as sd
import matrixmaker as mm
import pandas as pd
import numpy as np

i = os.getenv('SLURM_ARRAY_TASK_ID')
# if i!>0:
# i=1
print(i)


fbands=1/2**np.arange(1,11, 1)[:0:-1]
per=1/fbands
print(per)
matureagematch=np.int64(per[-2:0:-1])

mm.frequency_phaseplane(matureage=matureagematch,
fbands=fbands,
numberofdays=1001, bhmax=.3, driftmax=.03,
strategy_resolution=21,
numberofbins=100,
envivariance=[.2],
filename_prefix="FreqvsMatureAge_1001days_.2ev_.1emv_R"+str(i),
envimeanvariance=[.1],
savepath="../Results/",
savealldays=True,
birthrate=[40],
i=i
)