import simpledrift as sd
import matrixmaker as mm
import numpy as np
import imp
import matplotlib.pyplot as plt
import os
i = os.getenv('SLURM_ARRAY_TASK_ID')
# if i!>0:
# i=1

if not isinstance(i, int):
    i=1

print(i)


bhlower=0
bhupper=.5
bhinterval=10
driftlower=0
driftupper=.1
driftinterval=10

numberofbins=100
numberofdays=50
envimeanvariance=.1
envivariance=.3

# lowerbound=.1
# upperbound=.5
power=0
# fsp='$SCRATCH/debivort_lab/Ryan/figs'
fsp='figs'
oversamplerate=10
# geommean=np.ones([bhinterval,driftinterval])
# arithmean=np.zeros([bhinterval,driftinterval])

numsims=1
numbands=4

bandexponent=2.0
bands=.5*(((np.arange(numbands+1))/numbands))
bands=np.arange(numbands+1)**bandexponent
bands=np.arange(numbands+1)**bandexponent
bands/=np.max(bands)/.5

bands=[0,1/32,1/16,1/8,1/4,1/2]

for f in range(len(bands)-1):
    lowerbound=bands[f]
    upperbound=bands[f+1]
    # for i in range(numsims):
        # display("Simulation:"+str(i))
    rw=sd.makefilterednoise(numberofdays=numberofdays, envimeanvariance=envimeanvariance, envivariance=envivariance, power=0, lowerbound=lowerbound, upperbound=upperbound, oversamplerate=oversamplerate)
        # fig,ax=plt.subplots()
        # c=ax.pcolormesh(rw)
        # ax.set_xlabel('Day')
        # ax.set_ylabel('Preference')
        # fig.colorbar(c, ax=ax)
        # plt.show()

    mm.matrixmaker(rw, bhlower, bhupper, bhinterval, driftlower, driftupper, driftinterval, runindex=i, fband=f, envimeanvariance=envimeanvariance, envivariance=envivariance, power=power, freqmax=upperbound, freqmin=lowerbound)
        # print(matrix)

# for i in range(numsims):
rw=sd.makefilterednoise(numberofdays=numberofdays, envimeanvariance=envimeanvariance, envivariance=envivariance, power=0, oversamplerate=oversamplerate)

mm.matrixmaker(rw, bhlower, bhupper, bhinterval, driftlower, driftupper, driftinterval, runindex=i, fband='W', envimeanvariance=envimeanvariance, envivariance=envivariance, power=power)
rw=sd.makefilterednoise(numberofdays=numberofdays, envimeanvariance=envimeanvariance, envivariance=envivariance, power=1, oversamplerate=oversamplerate)

mm.matrixmaker(rw, bhlower, bhupper, bhinterval, driftlower, driftupper, driftinterval, runindex=i, fband='P', envimeanvariance=envimeanvariance, envivariance=envivariance, power=1)

#     geommean*=matrix
#     arithmean+=matrix
# arithmean/=numsims
# geommean=np.power(geommean, 1/numsims)
