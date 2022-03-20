import simpledrift as sd
import matrixmaker as mm
import numpy as np
# import imp
# import matplotlib.pyplot as plt
import os
# import math
import xarray
from datetime import date

i = os.getenv('SLURM_ARRAY_TASK_ID')
# i=1
# bands=[0,1/32,1/16,1/8,1/4,1/2]

fres=21
phaseplane_res=21
fres=21
strategy_res=51
control_res=3

savepath="../Results/figs_1-7-21/"

fbands=np.arange(21)

fbands=fbands[:0:-1]
# fbands=np.array([4.0])
fbands=2*fbands
bands=1/fbands
bands.round(3)

# Each run should run all fbands, each iteration is one run 

# output3=mm.matrixmaker(sa, bhstrats=np.linspace(0, .5, 51), driftstrats=np.linspace(0, .1, 51),
#                       envimeanvariance=np.linspace(.06, .3, 5), envivariance=np.linspace(.06, .3, 5), birthrate=np.linspace(20,60,3, dtype=int), matureage=np.linspace(5,20,4, dtype=int), nameprefix="test3")

first_iteration=True
#Step 1: Make the core signals
for f in bands:
    print(str("F"+str(f)+"_R"+str(i)))
    envi, us=sd.makefilterednoise(lowerbound=f/2, upperbound=f*2, numberofdays=101)
    dataframe=mm.matrixmaker(us, bhstrats=np.linspace(0, .5, strategy_res),
    driftstrats=np.linspace(0, .1, strategy_res),
    envimeanvariance=np.linspace(.06, .3, 5),
    figuresavepath=savepath,
    envivariance=np.linspace(.1, .3, 3),
    # envivariance=[.2],
    # birthrate=np.linspace(20,60,3, dtype=int),
    birthrate=[20],
    savealldays=False,
    matureage=np.linspace(5,20,4, dtype=int),
    # matureage=[10],
    nameprefix=str("F"+str(np.round(f, 3))),
    runindex=i,
    savedata=False
    )
    
    if first_iteration:
        x_all=dataframe.to_xarray()
        x_all=x_all.expand_dims({"freq":[f]})
        first_iteration=False
    else:
        x=dataframe.to_xarray()
    # x_all=x_all+x #But actual xarray concatenation
        # x.merge(x_all, join=outer)
        x=x.expand_dims({"freq":[f]})
        x_all=xarray.concat([x_all, x], dim="freq")
        # print('merged')
        # x_all=xarray.merge([x_all, x])

# Step 2, collate each run into one xarray
# print(x_all)
# savepath="../Results/figs/"
today = date.today()
filename=savepath+str(today)+"_R"+str(i)+".nc"
x_all.to_netcdf(filename)