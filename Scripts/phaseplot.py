import simpledrift as sd
import matrixmaker as mm
import numpy as np
import imp
import matplotlib.pyplot as plt
import os
import math

# import tracemalloc

# tracemalloc.start()


i = os.getenv('SLURM_ARRAY_TASK_ID')
# if i!>0:
# i=1
print(i)

# If not running as part of a batch, then just set i=1
# if not isinstance(i, int):
#     i=1 

i=int(i)
print(i)

#Parameters for driftvsbethedging matrix
bhlower=0
bhupper=.5
bhinterval=10
driftlower=0
driftupper=.1
driftinterval=10

#Parameters for each trial
numberofbins=100
numberofdays=120

# Parameters for generating stimuli
# Currently set to 

# lowerbound=.1
# upperbound=.5
power=0
# fsp='$SCRATCH/debivort_lab/Ryan/figs'
fsp='../Results/figs'
oversamplerate=10
lengthbuffer=1000
# geommean=np.ones([bhinterval,driftinterval])
# arithmean=np.zeros([bhinterval,driftinterval])

numsims=1
# numbands=4

# bandexponent=2.0
# bands=.5*(((np.arange(numbands+1))/numbands))
# bands=np.arange(numbands+1)**bandexponent
# bands=np.arange(numbands+1)**bandexponent
# bands/=np.max(bands)/.5

bands=[0,1/32,1/16,1/8,1/4,1/2]

fbands=np.arange(17)
fbands=fbands[:0:-1]
fbands=2*fbands
# print(fx)
bands=1/fbands

# mabands=[0,2,5,10,20]
# mabands=np.arange()
# mabands=np.arange(16)*2
# mabands=np.arange(8)*2
mabands=[5,10]

brbands=[10]

mod_enviMeanVariance=15
emvstepsize=.02
mod_enviVariance=1
mod_fbands=len(bands)+1
mod_mabands=len(mabands)
mod_brbands=len(brbands)

totalcombinations=mod_enviMeanVariance*mod_enviVariance*mod_fbands*mod_mabands*mod_brbands

cm=mod_enviMeanVariance
cd=1 #Common denominator, for calculating values based on bash index

# envimeanvariance=.1

i_enviMeanVariance=math.floor(i%cm/cd)
envimeanvariance=emvstepsize+(i_enviMeanVariance)*emvstepsize

cd=cm
cm*=mod_enviVariance
i_envivariance=math.floor(i%cm/cd)
envivariance=.1+i_envivariance*.1

cd=cm
cm*=mod_fbands
f=math.floor(i%cm/cd)

cd=cm
cm*=mod_mabands
m=math.floor(i%cm/cd)

cd=cm
cm*=mod_brbands
br=math.floor(i%cm/cd)

if f<len(bands)-1:
    lowerbound=bands[f]
    upperbound=bands[f+1]

    rw=sd.makefilterednoise(numberofdays=numberofdays, envimeanvariance=envimeanvariance, envivariance=envivariance, power=0, lowerbound=lowerbound, upperbound=upperbound,
            lengthbuffer=lengthbuffer, oversamplerate=oversamplerate)
        # fig,ax=plt.subplots()
        # c=ax.pcolormesh(rw)
        # ax.set_xlabel('Day')
        # ax.set_ylabel('Preference')
        # fig.colorbar(c, ax=ax)
        # plt.show()
    for m in range(len(mabands)):
        for br in range(len(brbands)):
            mm.matrixmaker(rw, bhlower, bhupper, bhinterval, driftlower, driftupper, driftinterval, runindex=i, fband=f, envimeanvariance=envimeanvariance, envivariance=envivariance, power=power, freqmax=upperbound, freqmin=lowerbound, birthrate=brbands[br],
             matureage=mabands[m])
