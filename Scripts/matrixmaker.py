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
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import xarray as xr
from datetime import date
from sys import getsizeof

def matrixmaker(envi, bhstrats=np.linspace(0, 1, 4), driftstrats=np.linspace(0, 1, 4), showgraphs=False, figuresavepath='../Results/figs',
     runindex=0, environtype=-1, freqmin=-1, freqmax=-1, power=-1, envimeanvariance=[.3], envivariance=[.1], birthrate=[40], matureage=[10],
     numberofbins=100,
     driftmaxdistribution=.3,
     nameprefix="", savealldays=True, savedata=True, saveenv=True):

    prefmean=0
    adaptivetracking=0


    iterables=[matureage, np.round(birthrate,3), np.round(envivariance,3), np.round(envimeanvariance,3), np.round(bhstrats,3), np.round(driftstrats,3)]
#     interables=np.round(iterables,3)
#     print(iterables)
    index=pd.MultiIndex.from_product(iterables, names=["matureage", "birthrate", "envvar", "envmeanvar", "bet-hedging", "drift", ])
    # print(index)

    allseries=Parallel(n_jobs=-1, verbose=5)(delayed(sd.driftmodeling)(envi, prefmean,
        prefvariance=i[4],


        # prefvariance=0, #bh disabled to look at the effect of the bounding box

        driftvariance=i[5], 
        adaptivetracking=adaptivetracking, 
        birthrate=i[1],
        matureage=i[0],
        driftmaxdistribution=driftmaxdistribution,
        # driftmaxdistribution=0,

        showgraphs=showgraphs,
        figuresavepath=figuresavepath,
        envimeanvariance=i[3],
        envivariance=i[2],
        numberofbins=numberofbins,
        # envivariance=.001,
        savealldays=savealldays) for i in index)
            # print(["Drift is: ", driftvariance[y], "Bet-hedging is: ", driftvariance[x]] )
            # print(matrix)
    dataframe=pd.DataFrame(allseries, index=index, dtype='float')
    ## This version is for day names, rather than indexes
    # if savealldays:
    #     columnnames=np.empty(dataframe.shape[1], dtype='object')
    #     for i in range(dataframe.shape[1]):
    #         columnnames[i]='Day '+str(i)
    #     dataframe.columns=columnnames
    # else:
    #     columnnames=np.empty(dataframe.shape[1], dtype='object')
    #     for i in range(dataframe.shape[1]):
    #         columnnames[i]='Day '+str(envi.shape[0])
    #     dataframe.columns=columnnames 

    if savealldays:
        columnnames=np.empty(dataframe.shape[1], dtype='object')
        for i in range(dataframe.shape[1]):
            columnnames[i]=i
        dataframe.columns=columnnames
    else:
        columnnames=np.empty(dataframe.shape[1], dtype='object')
        for i in range(dataframe.shape[1]):
            columnnames[i]=envi.shape[0]-1
            print(i)
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
        # pqtname=filename+'_Populations.parquet'
        ncname=filename+"_Populations"
        dataframe=xr.DataArray(dataframe)
        dataframe=dataframe.unstack().rename({"dim_1":"day"})
        # dataframe=dataframe.unstack()
        if savedata:
            # print(ncname)
            # dataframe.to_parquet(path=os.path.join(figuresavepath,pqtname))

            dataframe.to_netcdf(ncname+".nc")
        dataframe=dataframe.unstack()
        
        enviname=filename+'_env'
        if saveenv:
            np.save(os.path.join(figuresavepath,enviname), envi)
#         print(figuresavepath)
    else:
        print('not saving, no valid path')
    print(getsizeof(dataframe))
    return dataframe

def frequency_phaseplane(i=1, fbands=1/(np.arange(1,21)[:0:-1]*2), strategy_resolution=51, numberofdays=101,
 matureage=[10], envimeanvariance=[.1], birthrate=[20], envivariance=[.1], bhmax=.5, driftmax=.1, numberofbins=100,
 savealldays=False,
 savepath="../Results/", filename_prefix=""):

    control_res=3

    # savepath="../Results/figs_1-7-21/"

    # fbands=np.arange(frequency_resolution)

    # fbands=fbands[:0:-1]
    # # fbands=np.array([4.0])
    # fbands=2*fbands
    # bands=1/fbands
    fbands.round(3)

    # Each run should run all fbands, each iteration is one run 

    # output3=mm.matrixmaker(sa, bhstrats=np.linspace(0, .5, 51), driftstrats=np.linspace(0, .1, 51),
    #                       envimeanvariance=np.linspace(.06, .3, 5), envivariance=np.linspace(.06, .3, 5), birthrate=np.linspace(20,60,3, dtype=int), matureage=np.linspace(5,20,4, dtype=int), nameprefix="test3")

    first_iteration=True
    #Step 1: Make the core signals
    for fi in np.arange(1,len(fbands)-1):
        f=fbands[fi]
        print(str("F"+str(f)+"_R"+str(i)))
        envi, us=sd.makefilterednoise(lowerbound=fbands[fi-1], upperbound=fbands[fi+1], numberofdays=numberofdays)

        dataframe=matrixmaker(us, bhstrats=np.linspace(0, bhmax, strategy_resolution),
        driftstrats=np.linspace(0, driftmax, strategy_resolution),
        numberofbins=numberofbins,
        figuresavepath=savepath,
        # envivariance=np.linspace(.1, .3, 3),
        envimeanvariance=envimeanvariance,
        envivariance=envivariance,
        # birthrate=np.linspace(20,60,3, dtype=int),
        birthrate=birthrate,
        savealldays=savealldays,
        matureage=matureage,
        # matureage=[10],
        nameprefix=str("F"+str(np.round(f, 3))),
        runindex=i,
        savedata=False
        )
        # print(dataframe)


        #make environment into array that can be merged into xarray total
        
        envda=xr.DataArray(np.expand_dims(us, 1), coords={"day": np.arange(numberofdays), "freq":[f]}, dims=["day", "freq"], name="envi")
        envda.swap_dims()
        print(envda)
        if first_iteration:
            if len(dataframe.shape)<3:
                x_all=xr.DataArray(dataframe, dims=("MultiIndex", "day")).unstack("MultiIndex")
            else:
                x_all=dataframe

            # x_all=dataframe.to_xarray()
            x_all=x_all.expand_dims({"freq":[f]})
            first_iteration=False

            x_all=xr.Dataset({"environment_mean":envda, "sim_results":x_all})
        else:
            if len(dataframe.shape)<3:
                x=xr.DataArray(dataframe, dims=("MultiIndex", "day")).unstack("MultiIndex")
            else:
                x=dataframe
        # x_all=x_all+x #But actual xarray concatenation
            # x.merge(x_all, join=outer)
            x=x.expand_dims({"freq":[f]})
            first_iteration=False
            print(envda)
            x=xr.Dataset({"environment_mean":envda, "sim_results":x})
            x_all=xr.merge([x_all, x])
                        # x_all=xarray.merge([x_all, x], dim="freq")

            # print('merged')conda
            # x_all=xarray.merge([x_all, x])

    # Step 2, collate each run into one xarray
    # print(x_all)
    # savepath="../Results/figs/"
    today = date.today()
    filename=savepath+filename_prefix+str(today)+"_R"+str(i)+".nc"
    print(f"Run saved at {filename}")
    x_all.to_netcdf(filename)
    print(getsizeof(x_all))
    return x_all

def getallruns(listofinputs, day=1000):
    for i, input in enumerate(listofinputs):
        print(i,input)
        sin=xr.open_dataset(input)
        sin.coords["Run"]=i
        sin=sin.expand_dims("Run")
        sin=sin.isel(day=day)
        if i==0:
            allruns=sin
        else:
            allruns=allruns.merge(sin)
    return allruns

def calcgeomean(allruns):
    geomean=np.log10(allruns['sim_results'].squeeze()).mean(dim="Run")
    return geomean  


def makeMatureAgexFreqFigure2(phaseplaneoutput):

    for j, freq in enumerate(np.array(phaseplaneoutput['freq'])):
        envi=sd.meantofullenvi(envimeans=phaseplaneoutput["environment_mean"][:, j], envimeanvariance=float(phaseplaneoutput["envmeanvar"]), envivariance=float(phaseplaneoutput["envvar"]), numberofbins=100) 
        fig=px.imshow(envi, aspect='auto')
        fig.show()
    for i in range(0,len(phaseplaneoutput['matureage'])):

        fig=px.imshow(np.log(phaseplaneoutput["sim_results"].isel({"day":100, "matureage":i}).squeeze()), facet_col="freq")

        fig.layout.title.text="Mature Age: "+str(int(phaseplaneoutput["matureage"][i]))
        fig.layout.coloraxis.colorbar.title.text="Final Population (Log)"
        for j, freq in enumerate(np.array(phaseplaneoutput['freq'])):
            # print(freq)
            fig.layout.annotations[j]['text']= 'Period = %d' %int(1/freq)
        fig.show()

def makeMatureAgexFreqFigure(phaseplaneoutput):

    gap=.005
    fig = make_subplots(rows=8, cols=7, shared_yaxes=True, vertical_spacing=gap, horizontal_spacing=gap)

    for j, freq in enumerate(np.array(phaseplaneoutput['freq'])):
        envi=sd.meantofullenvi(envimeans=phaseplaneoutput["environment_mean"][:, j], envimeanvariance=float(phaseplaneoutput["envmeanvar"]), envivariance=float(phaseplaneoutput["envvar"]), numberofbins=100) 
        fig.add_trace(go.Heatmap(z=envi), row=1, col=j+1)
        # fig.show()
        for i in range(0,len(phaseplaneoutput['matureage'])):

            fig.add_trace(go.Heatmap(z=np.log(phaseplaneoutput["sim_results"].isel({"day":100, "matureage":i, "freq":j}).squeeze())), col=j+1, row=i+2)
            fig.layout.coloraxis.colorbar.showticklabels=False
            # fig.layout.title.text="Mature Age: "+str(int(phaseplaneoutput["matureage"][i]))
            # fig.layout.coloraxis.colorbar.title.text="Final Population (Log)"
            # for j, freq in enumerate(np.array(phaseplaneoutput['freq'])):
            #     # print(freq)
            #     fig.layout.annotations[j]['text']= 'Period = %d' %int(1/freq)
    fig.update_layout(height=1200, width=1200)
    for anno in fig['layout']['annotations']:
        anno['text']=""
    for axis in fig.layout:
        
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
            fig.layout[axis].tickfont = dict(color = 'rgba(0,0,0,0)')
            
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''
            fig.layout[axis].tickfont = dict(color = 'rgba(0,0,0,0)')
        if type(fig.layout[axis]) == go.layout.coloraxis:
            fig.layout[axis].title.text = ''
            fig.layout[axis].tickfont = dict(color = 'rgba(0,0,0,0)')
    fig.show()
    fig.write_image('../Results/figs/MatureAgexFreq.pdf')

def dailychangehistogram(phaseplaneoutput):
    return phaseplaneoutput


def realworlddataPhaseSpace(realworlddata, i=1, strategy_resolution=51, numberofdays=101,
 matureage=[10], envimeanvariance=[.1], birthrate=[20], envivariance=[.1], bhmax=.5, driftmax=.1, numberofbins=100,
 savealldays=False,
 savepath="../Results/", filename_prefix=""):
    f=0
    first_iteration=True
    #Step 1: Make the core signals
    try:
        realworlddata=realworlddata.expand_dims("siteID")
    except:
        print("Processing multiple sites")
    try:
        realworlddata=realworlddata.expand_dims("variable")
    except:
        print("Processing Multiple Variables")
    try:
        realworlddata=realworlddata.expand_dims("chunk")
    except:
        print("Processing multiple Chunks")


    for ii, i in enumerate(realworlddata['siteID']):
        # print(i)
        for ji, j in enumerate(realworlddata['variable']):
            for ki, k in enumerate(realworlddata['chunk']):
                # print(k)  
                # 
                if ~np.isnan(realworlddata[ii,ji,ki]).all():
                    l=realworlddata[ii,ji,ki].copy()

                    siteID=i["siteID"].values[()]
                    variable=j["variable"].values[()]
                    chunk=k["chunk"].values[()]
                    # print(siteID, variable, chunk)
                    # print(l)
                    # print(i)
                    # print(j)
                    # print(k)
                # f=fbands[fi]
        # print(str("F"+str(f)+"_R"+str(i)))
        # envi, us=sd.makefilterednoise(lowerbound=fbands[fi-1], upperbound=fbands[fi+1], numberofdays=numberofdays)
                    # return l.values
                    # l[0]=0
                    #Run Simulations
                    dataframe=matrixmaker(l.values, bhstrats=np.linspace(0, bhmax, strategy_resolution),
                    driftstrats=np.linspace(0, driftmax, strategy_resolution),
                    numberofbins=numberofbins,
                    figuresavepath=savepath,
                    # envivariance=np.linspace(.1, .3, 3),
                    envimeanvariance=envimeanvariance,
                    envivariance=envivariance,
                    # birthrate=np.linspace(20,60,3, dtype=int),
                    birthrate=birthrate,
                    savealldays=savealldays,
                    matureage=matureage,
                    saveenv=False,
                    # matureage=[10],
                    # nameprefix=str("F"+str(np.round(f, 3))),
                    runindex=0,
                    savedata=False
                    )
                    # print(dataframe)


                    #make environment into array that can be merged into xarray total
                    # print(l)
                    # envda=xr.DataArray(np.expand_dims(l, 1), coords={"day": np.arange(numberofdays), "siteID":siteID, "chunk":chunk, "variable":variable}, dims=["day", "siteID", "chunk", "variable"], name="envi")
                    envda=l
                    envda.swap_dims()
                    if first_iteration:
                        x_all=xr.DataArray(dataframe, dims=("MultiIndex", "day")).unstack("MultiIndex")

                        # x_all=dataframe.to_xarray()
                        # print(x_all)
                        # x_all=x_all.expand_dims(coords={"day": np.arange(numberofdays),"siteID":siteID, "chunk":chunk, "variable":variable}, dims=["day", "siteID", "chunk", "variable"])
                        first_iteration=False

                        x_all=xr.Dataset({"environment_mean":envda, "sim_results":x_all})
                        x_all=x_all.expand_dims(dim=["siteID", "variable", "chunk"])
                    else:
                        x=xr.DataArray(dataframe, dims=("MultiIndex", "day")).unstack("MultiIndex")
                    # x_all=x_all+x #But actual xarray concatenation
                        # x.merge(x_all, join=outer)
                        # x=x.expand_dims({"siteID":siteID, "chunk":chunk, "variable":variable})
                        first_iteration=False
                        x=xr.Dataset({"environment_mean":envda, "sim_results":x})
                        x=x.expand_dims(dim=["siteID", "variable", "chunk"])
                        # print(x_all)
                        # print(x)
                        try:
                            x_all=xr.combine_by_coords([x_all, x])
                            f+=1
                            # print(f)
                        except:
                            try:
                                x_all=xr.concat([x_all,x], dim="siteID")
                            except:
                                x_all=xr.concat([x_all,x], dim="variable")
                            # print("bug trying to combine runs")
                            # return x_all, x

                        # x_all=xr.merge([x_all, x])
                                    # x_all=xarray.merge([x_all, x], dim="freq")
        #                 break
        #         else:
        #             continue
        #         break
        #     else:
        #         continue
        #     break
        # else:
        #     continue
        # break
               
                        


                        # print('merged')conda
                        # x_all=xarray.merge([x_all, x])

    # Step 2, collate each run into one xarray
    # print(x_all)
    # savepath="../Results/figs/"
    today = date.today()
    filename=savepath+filename_prefix+str(today)+"_R"+str(i)+".nc"
    # x_all.to_netcdf(filename)
    return x_all