# make the matrix
# call simpledrift to fill in the values for different drift and bh
# pass on to heatmap
# pay attention to axes so they give the correct values
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
import simpledrift as sd
import math
from joblib import Parallel, delayed
import os
import pandas as pd

def matrixmaker(envi, bhstrats=np.linspace(0, 1, 4), driftstrats=np.linspace(0, 1, 4), showgraphs=False, figuresavepath='../Results/figs',
     runindex=0, environtype=-1, freqmin=-1, freqmax=-1, power=-1, envimeanvariance=[.1], envivariance=[.1], birthrate=[10], matureage=[10],
     nameprefix="", savealldays=True, savedata=True, saveenv=True):

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
#     bhstrats=np.linspace(bhlower, bhupper, bhinterval)
#     driftstrats=np.linspace(driftlower, driftupper, driftinterval)
# 
#     numruns=bhinterval*driftinterval

    #testing values are right
    # for n in range(numruns):
    #     print('driftvariance'+str(driftvariance[math.floor(n/bhinterval)])+ 'prefvariance'+ str(prefvariance[n%bhinterval]))

#     flatmatrix=np.zeros(numruns)

    # for x in range(bhinterval):

    iterables=[matureage, np.round(birthrate,3), np.round(envivariance,3), np.round(envimeanvariance,3), np.round(bhstrats,3), np.round(driftstrats,3)]
#     interables=np.round(iterables,3)
#     print(iterables)
    index=pd.MultiIndex.from_product(iterables, names=["matureage", "birthrate", "envvar", "envmeanvar", "bet-hedging", "drift", ])

    allseries=Parallel(n_jobs=-1, verbose=10)(delayed(sd.driftmodeling)(envi, prefmean,
        prefvariance=i[4], driftvariance=i[5], adaptivetracking=adaptivetracking, birthrate=i[1],matureage=i[0], showgraphs=showgraphs, figuresavepath=figuresavepath, envimeanvariance=i[3], envivariance=i[2], savealldays=savealldays) for i in index)
            # print(["Drift is: ", driftvariance[y], "Bet-hedging is: ", driftvariance[x]] )
            # print(matrix)
    dataframe=pd.DataFrame(allseries, index=index, dtype='float')
    if savealldays:
        columnnames=np.empty(dataframe.shape[1], dtype='object')
        for i in range(dataframe.shape[1]):
            columnnames[i]='Day '+str(i)
        dataframe.columns=columnnames
    else:
        columnnames=np.empty(dataframe.shape[1], dtype='object')
        for i in range(dataframe.shape[1]):
            columnnames[i]='Day '+str(envi.shape[0])
        dataframe.columns=columnnames 
    # matrix=np.zeros((bhinterval,driftinterval))
#     matrix[:,:]=flatmatrix.reshape((bhinterval, driftinterval), order='F')
#     matrixlog=np.log(matrix)
#     # print(matrixlog)
#     print(matrix)

    # bhmargin=(bhupper-bhlower)/(bhinterval-1)/2
#     driftmargin=(driftupper-driftlower)/(driftinterval-1)/2
#     prefvariancemesh=np.linspace(bhlower-bhmargin, bhupper+bhmargin, bhinterval+1)
#     driftvariancemesh=np.linspace(driftlower-driftmargin, driftupper+driftmargin, driftinterval+1)
#     # print(prefvariancemesh)
#     # print(driftvariancemesh)
#     # driftvariancegrid2, prefvariancegrid2= np.meshgrid(driftvariance, prefvariance)

    # print(matrix)

#     fig, (ax1, ax2)=plt.subplots(2, 1)
# 
#     fig.set_figwidth(10)
#     fig.set_figheight(12)
#     fig.tight_layout()
#     plt.subplots_adjust(hspace=.3)
#     ax1.set_xlabel('Day')
#     ax1.set_ylabel('Bin')
# 
#     if freqmin>=0:
#         ax1.set_title('Environment (Filtered to '+str(round(freqmin, 3))+' - '+str(round(freqmax,3))+' cycles/day')
#     else:
#         ax1.set_title('Environment')
# 
# 
#     b=ax1.pcolormesh(envi, cmap='Greys')
#     fig.colorbar(b, ax=ax1)
# 
# 
#     scale=max(-np.min(matrixlog),np.max(matrixlog))
#     c=ax2.pcolormesh(driftvariancemesh, prefvariancemesh, matrixlog, shading='flat', cmap='RdBu', vmin=-scale, vmax=scale)
#     
#     ax2.set_xlabel('Drift')
#     ax2.set_ylabel('Bet-Hedging')
#     fig.colorbar(c, ax=ax2)
#     ax2.set_title('Log of Final Population')
#     # plt.show()
# 
    if os.path.exists(figuresavepath):
#         heatmapname='R'+str(runindex)+'_heatmap.png'
#         filename='BR'+str(birthrate)+'MA'+str(matureage)+'R'+str(
#         runindex)+'_Env_FinalPopulations.npz'
        if nameprefix=="":
          filename='R'+str(runindex)
        else:
          filename=nameprefix+'_R'+str(runindex)
        
        if environtype!=-1:
          # heatmapname='BR'+str(birthrate)+'MA'+str(matureage)+'F'+str(environtype)+'_R'+str(runindex)+'_heatmap.pdf'
          filename+='_T'+str(environtype)+'Populations.parquet'
        # fig.savefig(os.path.join(figuresavepath,heatmapname),
#         bbox_inches='ti ght', pad_inches=.3)
#         np.savez(os.path.join(figuresavepath,filename),
#         finalpopulations=matrix, prefvariancemesh=prefvariancemesh,
#         driftvariancemesh=driftvariancemesh, envi=envi,
#         freqmin=freqmin, freqmax=freqmax, power=power,
#         envimeanvariance=envimeanvariance,
#         envivariance=envivariance, birthrate=birthrate,
#         matureage=matureage)
        pqtname=filename+'_Populations.parquet'
        if savedata:
            dataframe.to_parquet(path=os.path.join(figuresavepath,pqtname))
        
        enviname=filename+'_env'
        if saveenv:
            np.save(os.path.join(figuresavepath,enviname), envi)
#         print(figuresavepath)
    else:
        print('not saving, no valid path')

    return dataframe
