import numpy as np
import matplotlib.pyplot as plt
import os
import simpledrift as sd 
import scipy.stats as sci

def row_col_argmax(a):
    '''
    Given a 2-D array a,
    returns the row index + column index
    of the array's maximum
    from here: https://stackoverflow.com/a/9483964
    '''
    return np.unravel_index(a.argmax(), a.shape)

def environmentmapsmaker(dat, reverseindex, ind=0, excludelast=10):

  
  randrange=reverseindex.shape[0]-excludelast
  # randrange=1
  envis=dat['envifull']
  emv=dat['envimeanvariance']
  em=dat['envimean']
  fb=dat['fbands']
  
  fig, axs=plt.subplots(reverseindex.shape[1],reverseindex.shape[2])
  # fig, axs=plt.subplots(2,2, sharey=True, sharex=True)
  fig.set_size_inches(5*reverseindex.shape[1],5*reverseindex.shape[2]*1.2)

  # print(reverseindex.shape)
  bands=[0,1/32,1/16,1/8,1/4,1/2]
  # hm=np.log10(np.sum(reverseindex, axis=(2,3)))

  # hm=np.log10(reverseindex)
  # vmag=(np.max(hm))


  # hm=np.where(hm<-5, -1, hm)

  # print(hm[5,5,0,0])
  numberofbins=100
  numberofdays=50
  # envi=np.zeros((numberofbins,numberofdays))
  envi=np.zeros((numberofbins,numberofdays,reverseindex.shape[1],reverseindex.shape[2]))

  x=np.linspace(-1,1,numberofbins)
  envivariance=.3
  maxsurvivalrate=1

  for f in range(reverseindex.shape[1]):
      for g in range(reverseindex.shape[2]):
          enviindex=reverseindex[np.random.randint(0, randrange), f, g]
          print(f'Row {f}, Column {g}, Index: {enviindex}, F: {fb[enviindex]}, EMV: {emv[enviindex]}, EM: {em[enviindex][0]}')
          print(f'Biggest Envi is {np.max(envis[enviindex])}')
          # if fb[enviindex]=='P':
          #   p=1
          # else:
          #   p=0
          for t in range(numberofdays):
            envi[:,t, f, g]=sci.norm.pdf(x, em[enviindex][t]/50-1, envivariance) # A gaussian of environment with center around 0
            if np.max(envi[:,t, f, g])==0:
              print(f'err {em[enviindex][t]}')
              print(envi[:,t, f, g])
            envi[:,t, f, g]=envi[:,t, f, g]/np.max(envi[:,t, f, g])*maxsurvivalrate

          # # envi=sd.makefilterednoise(envimeanvariance=emv[enviindex], power=p)
          
          pcm=axs[f,g].pcolormesh(envi[:,:, f, g],
          # cmap='hot',
          # cmap='inferno'
          cmap='Greys'
          # cmap='viridis',

          # vmin=-1,
          # vmax=vmag
          )
  #             annot=True,
  #             linewidths=.1, linecolor='grey',
  #             cbar_kws = {'label': 'log_10 num simulations'},
  #             xticklabels=False,yticklabels=False,
              # cbar=False,
              # ax=axs[f,g])
          # axs[f,g].set_frame_on
          axs[f,g].set_xticklabels('')
          axs[f,g].set_yticklabels('') 



  fig.tight_layout()
  # plt.subplots_adjust(hspace=.6)

  plt.subplots_adjust(bottom=0.1, right=0.95, top=0.9)
  cax = plt.axes([0.96, 0.785, 0.01, 0.115])
  plt.colorbar(pcm, cax=cax)

  fig.savefig(os.path.join('randenv_'+str(ind)+'.pdf'))
  print('done')

def plotspecificrun(dat, runnumber, figuresavepath='figs'):
  numberofdays=50
  numberofbins=100
  runindex=runnumber
  freqmin=0
  freqmax=0
  fband=-1

  envis=dat['envifull']
  emv=dat['envimeanvariance']
  em=dat['envimean']
  fb=dat['fbands']

  envi=np.zeros([numberofbins,numberofdays])
  
  x=np.linspace(-1,1,numberofbins)
  envivariance=.3
  maxsurvivalrate=1

  for t in range(numberofdays):
    envi[:,t]=sci.norm.pdf(x, em[runnumber][t]/50-1, envivariance) # A gaussian of environment with center around 0
    # if np.max(envi[:,t, f, g])==0:
    #   print(f'err {em[runnumber][t]}')
    #   # print(envi[:,t, f, g])
    envi[:,t]=envi[:,t]/np.max(envi[:,t])*maxsurvivalrate

  fig, (ax1, ax2)=plt.subplots(2, 1)
  fig.set_figwidth(10)
  fig.set_figheight(12)
  fig.tight_layout()
  plt.subplots_adjust(hspace=.3)
  ax1.set_xlabel('Day')
  ax1.set_ylabel('Bin')

  # if freqmin>=0:
  #     ax1.set_title('Environment (Filtered to '+str(round(freqmin, 3))+' - '+str(round(freqmax,3))+' cycles/day')
  # else:
  ax1.set_title('Environment')


  b=ax1.pcolormesh(envi, cmap='Greys')
  fig.colorbar(b, ax=ax1)

  matrixlog=np.log10(dat['finalpopulations'][runnumber])
  driftvariancemesh=dat['x_driftvariancemesh']
  prefvariancemesh=dat['y_prefvariancemesh']

  scale=max(-np.min(matrixlog),np.max(matrixlog))
  c=ax2.pcolormesh(driftvariancemesh, prefvariancemesh, matrixlog, shading='flat', cmap='RdBu', vmin=-scale, vmax=scale)
  
  ax2.set_xlabel('Drift')
  ax2.set_ylabel('Bet-Hedging')
  fig.colorbar(c, ax=ax2)
  ax2.set_title('Log of Final Population')
  plt.show()

  if os.path.exists(figuresavepath):
      heatmapname='rR'+str(runindex)+'_heatmap.pdf'
      # filename='R'+str(runindex)+'_Env_FinalPopulations.npz'

      if fband!=-1:
          heatmapname='F'+str(fband)+'_R'+str(runindex)+'_heatmap.pdf'
          # filename='F'+str(fband)+'_R'+str(runindex)+'_Env_FinalPopulations.npz'

      fig.savefig(os.path.join(figuresavepath,heatmapname),bbox_inches='tight', pad_inches=.3)
      # np.savez(os.path.join(figuresavepath,filename), finalpopulations=matrix, prefvariancemesh=prefvariancemesh, driftvariancemesh=driftvariancemesh, envi=envi, freqmin=freqmin, freqmax=freqmax, power=power, envimeanvariance=envimeanvariance, envivariance=envivariance)
      print(figuresavepath)
  else:
      print('not saving, no valid path')

  # return matrix, driftvariance, prefvariance
