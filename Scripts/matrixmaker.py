# make the matrix
# call simpledrift to fill in the values for different drift and bh
# pass on to heatmap
# pay attention to axes so they give the correct values
import numpy as np
import matplotlib.pyplot as plt
import simpledrift as sd
import math
from joblib import Parallel, delayed
import os

def matrixmaker(envi, bhlower, bhupper, bhinterval, driftlower, driftupper, driftinterval, showgraphs=False, figuresavepath='../Results/figs',
     runindex=0, fband=-1, freqmin=-1, freqmax=-1, power=-1, envimeanvariance=-1, envivariance=-1, birthrate=10, matureage=10):

    # flynum=1
    # numberofbins=100
    # numberofdays=100
    # envimean=0
    # envivariance=.25
    # maxsurvivalrate=1
    # gain=.4
    # per=20
    # envi=sd.sinwaveinput(numberofbins, numberofdays, envimean, envivariance, maxsurvivalrate, gain, per)
    #

    prefmean=0
    # birthrate=10
    # matureage=10
    percentbh=0.01
    adaptivetracking=0
    # intervals=16

    # figuresavepath='figs'
    prefvariance=np.linspace(bhlower, bhupper, bhinterval)
    driftvariance=np.linspace(driftlower, driftupper, driftinterval)

    numruns=bhinterval*driftinterval

    #testing values are right
    # for n in range(numruns):
    #     print('driftvariance'+str(driftvariance[math.floor(n/bhinterval)])+ 'prefvariance'+ str(prefvariance[n%bhinterval]))


    flatmatrix=np.zeros(numruns)

    # for x in range(bhinterval):
    flatmatrix[:]=Parallel(n_jobs=-1, verbose=10)(delayed(sd.driftmodeling)(envi, prefmean,
        [prefvariance[n%bhinterval]], [driftvariance[math.floor(n/bhinterval)]], adaptivetracking, birthrate,matureage, percentbh, showgraphs, figuresavepath) for n in range(numruns))
            # print(["Drift is: ", driftvariance[y], "Bet-hedging is: ", driftvariance[x]] )
            # print(matrix)
    # print(flatmatrix)

    # flatmatrix=np.matrix(flatmatrix)
    # flatmatrix=
    matrix=np.zeros((bhinterval,driftinterval))
    matrix[:,:]=flatmatrix.reshape((bhinterval, driftinterval), order='F')
    matrixlog=np.log(matrix)
    # print(matrixlog)
    print(matrix)



    bhmargin=(bhupper-bhlower)/(bhinterval-1)/2
    driftmargin=(driftupper-driftlower)/(driftinterval-1)/2
    prefvariancemesh=np.linspace(bhlower-bhmargin, bhupper+bhmargin, bhinterval+1)
    driftvariancemesh=np.linspace(driftlower-driftmargin, driftupper+driftmargin, driftinterval+1)
    # print(prefvariancemesh)
    # print(driftvariancemesh)
    # driftvariancegrid2, prefvariancegrid2= np.meshgrid(driftvariance, prefvariance)

    # print(matrix)

    fig, (ax1, ax2)=plt.subplots(2, 1)

    fig.set_figwidth(10)
    fig.set_figheight(12)
    fig.tight_layout()
    plt.subplots_adjust(hspace=.3)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Bin')

    if freqmin>=0:
        ax1.set_title('Environment (Filtered to '+str(round(freqmin, 3))+' - '+str(round(freqmax,3))+' cycles/day')
    else:
        ax1.set_title('Environment')


    b=ax1.pcolormesh(envi, cmap='Greys')
    fig.colorbar(b, ax=ax1)


    scale=max(-np.min(matrixlog),np.max(matrixlog))
    c=ax2.pcolormesh(driftvariancemesh, prefvariancemesh, matrixlog, shading='flat', cmap='RdBu', vmin=-scale, vmax=scale)
    
    ax2.set_xlabel('Drift')
    ax2.set_ylabel('Bet-Hedging')
    fig.colorbar(c, ax=ax2)
    ax2.set_title('Log of Final Population')
    # plt.show()

    if os.path.exists(figuresavepath):
        heatmapname='R'+str(runindex)+'_heatmap.png'
        filename='BR'+str(birthrate)+'MA'+str(matureage)+'R'+str(runindex)+'_Env_FinalPopulations.npz'

        if fband!=-1:
            heatmapname='BR'+str(birthrate)+'MA'+str(matureage)+'F'+str(fband)+'_R'+str(runindex)+'_heatmap.pdf'
            filename='BR'+str(birthrate)+'MA'+str(matureage)+'F'+str(fband)+'_R'+str(runindex)+'_Env_FinalPopulations.npz'
        fig.savefig(os.path.join(figuresavepath,heatmapname),bbox_inches='tight', pad_inches=.3)
        np.savez(os.path.join(figuresavepath,filename),
            finalpopulations=matrix,
            prefvariancemesh=prefvariancemesh,
            driftvariancemesh=driftvariancemesh,
            envi=envi, freqmin=freqmin,
            freqmax=freqmax, power=power,
            envimeanvariance=envimeanvariance,
            envivariance=envivariance, birthrate=birthrate, matureage=matureage)
        print(figuresavepath)
    else:
        print('not saving, no valid path')

    return matrix, driftvariance, prefvariance
