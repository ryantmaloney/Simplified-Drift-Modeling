import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt
import time

def driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance, envimean, envivariance, driftvariance, gain, per, deathrate, birthrate, matureage):
    x=np.linspace(-1,1,numberofbins)
    maxage=30
    pref=np.zeros((numberofbins,numberofdays,maxage))
    reducebethedge=np.zeros((numberofbins,numberofdays,maxage))
    for q in range(0,2):
        pref[:,0,0]=sci.norm.pdf(x,prefmean[q],prefvariance[q]) # A gaussian of preference with center around 0
        pref[:,0,0]=pref[:,0,0]/np.sum(pref[:,0,0])*flynum # total # of flies=flynum
        # print(pref[:,0,0])
        reducebethedge[:,0,0]=sci.norm.pdf(x,np.multiply(prefmean[q],0.9),np.multiply(prefvariance[q],0.9)) # A gaussian of preference with center around 0
        reducebethedge[:,0,0]=reducebethedge[:,0,0]/np.sum(reducebethedge[:,0,0])*flynum # total # of flies=flynum
        # print(reducebethedge)
        global drift
        global bh

        envi=np.zeros((numberofbins,numberofdays))
        envi[:,0]=sci.norm.pdf(x,envimean,envivariance) # A gaussian of environment with center around 0
        envi=envi/(np.max(envi))*deathrate
        driftadvantage=np.zeros((numberofdays))
        blur=np.zeros((numberofbins,numberofbins))
        
        for b in range(numberofbins):
            blur[b,:]=sci.norm.pdf(x,x[b],driftvariance[q])
        for t in range(1,numberofdays):
            for w in range(2):
                if w==1:
                    pref[:,t,0]=reducebethedge[:,0,0]*birthrate/flynum*np.sum(pref[:,t-1,matureage:])
                    bh=pref[:,t,0]
                    #print(bh)
                    #print(pref[:,t,0])
                if w==2:
                    pref[:,t,0]=pref[:,0,0]*birthrate/flynum*np.sum(pref[:,t-1,matureage:])
                    drift=pref[:,t,0]
                    #print(drift)
                    #print(pref[:,t,0])

                envi[:,t]=sci.norm.pdf(x,(envimean+gain*np.sin(t*np.pi*2/per)),envivariance)
                envi[:,t]=envi[:,t]/np.max(envi[:,t])*.95

                for a in range(maxage):
                    driftadvantage[t]+=np.sum(np.multiply(pref[:,t-1,a], envi[:,t]))
                    #print(driftadvantage[t])

                for a in range(maxage):
                    pref[:,t,a]+=pref[:,t-1,a]
                    for b in range(numberofbins):
                        pref[:,t,a]+=pref[b,t-1,a]*blur[b,:]/np.sum(blur[b,:])
                        #pref[:,t,a]+=pref[b,t-1,a]*sci.norm.pdf(x,x[b],driftvariance)/np.sum(sci.norm.pdf(x,x[b],driftvariance))
                    pref[:,t,a]=np.multiply(pref[:,t,a], envi[:,t]) # Multiplying the preference to the environment       
            driftadvantage[t]=np.sum(pref[:,t,:])-driftadvantage[t]
            pref[:,t,1:]=pref[:,t,:-1]
            # print(bh)
            # print(drift)
 
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1)
        fig.set_figwidth(8)
        fig.set_figheight(8)
        fig.tight_layout()
        plt.subplots_adjust(hspace=.6)
        c=ax0.pcolormesh(envi)
        fig.colorbar(c,ax=ax0)
        ax0.set_title('Environment (color is fraction of flies of given pref survive)')
        ax0.set_ylabel('Preference')
        ax0.set_xlabel('Day')

        c=ax1.pcolormesh(np.sum(pref[:,:,2:],axis=2))
        fig.colorbar(c,ax=ax1)
        ax1.set_title('Fly Preference (color is log(num) flies each day)')
        ax1.set_ylabel('Preference')
        ax1.set_xlabel('Day')

        ax2.plot(np.log(np.sum(pref,axis=(0,2))))
        ax2.set_title('total log(num) flies)')
        ax2.set_ylabel('log(num) flies)')
        ax2.set_xlabel('Day')
        ax2.set_xlim(0,numberofdays)

        ax3.plot(driftadvantage/np.sum(pref,axis=(0,2)))
        ax3.set_title('Change in death rate due to drift ')
        ax3.set_ylabel('∆surviving flies/total flies')
        ax3.set_xlabel('Day')
        ax3.set_xlim(0,numberofdays)

        fig.colorbar(c,ax=ax2)
        fig.colorbar(c,ax=ax3)

        fig.suptitle('Bet-hedge variance: '+str(prefvariance[q])+', Drift variance: '+str(driftvariance[q]), y=-.05, fontsize=16)

        plt.show()