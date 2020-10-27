import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt

def driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance, envimean, envivariance, driftvariance, gain, per, deathrate, birthrate, matureage):
    x=np.linspace(-1,1,numberofbins)
    maxage=30
    pref=np.zeros((numberofbins,numberofdays,maxage))
    pref[:,0,0]=sci.norm.pdf(x,prefmean,prefvariance) # A gaussian of preference with center around 0
    pref[:,0,0]=pref[:,0,0]/np.sum(pref[:,0,0])*flynum # total # of flies=flynum
    #envi=gain*np.sin(x*2*np.pi/per+182*2*np.pi)+envimean
    # envi=gain*np.sin(x*2*np.pi/per+2*np.pi)+envimean
    # x=np.linspace(0, numberofdays, numberofbins)
    # envi=np.sin(x)+envimean
    # print(np.max(envi))
    envi=np.zeros((numberofbins,numberofdays))
    envi[:,0]=sci.norm.pdf(x,envimean,envivariance) # A gaussian of environment with center around 0
    # print(np.max(envi))
    envi=envi/(np.max(envi))*deathrate
    driftadvantage=np.zeros((numberofdays))
    for t in range(1,numberofdays):
        envi[:,t]=sci.norm.pdf(x,(envimean+gain*np.sin(t*np.pi*2/per)),envivariance)
        envi[:,t]=envi[:,t]/np.max(envi[:,t])*.95
        driftadvantage[t]=np.sum(np.multiply(pref[:,t-1], envi[:,t]))
        # print(driftadvantage[t])
        for b in range(numberofbins):
            for a in range(maxage):
                pref[:,t,a]+=pref[b,t-1,a]*sci.norm.pdf(x,x[b],driftvariance)/np.sum(sci.norm.pdf(x,x[b],driftvariance))
            # print(np.sum(sci.norm.pdf(x,x[b],driftvariance)))
        driftadvantage[t]=np.sum(np.multiply(pref[:,t], envi[:,t]))-driftadvantage[t]
        # plt.plot(pref[:,t])
        # plt.show()
        # print(pref[:,t])
        pref[:,t]=pref[:,t]+pref[:,0]*birthrate/flynum*np.sum(pref[:,t])
        pref[:,t]=np.multiply(pref[:,t], envi[:,t]) # Multiplying the preference to the environment
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

    c=ax1.pcolormesh(np.sum(pref,axis=2))
    fig.colorbar(c,ax=ax1)
    ax1.set_title('Fly Preference (color is log(num) flies each day)')
    ax1.set_ylabel('Preference')
    ax1.set_xlabel('Day')

    ax2.plot(np.log(np.sum(pref,axis=[1,2])))
    ax2.set_title('total log(num) flies)')
    ax2.set_ylabel('log(num) flies)')
    ax2.set_xlabel('Day')
    ax2.set_xlim(0,numberofdays)


    ax3.plot(driftadvantage/np.sum(pref,axis=[1,2]))
    ax3.set_title('Change in death rate due to drift ')
    ax3.set_ylabel('∆surviving flies/total flies')
    ax3.set_xlabel('Day')
    ax3.set_xlim(0,numberofdays)

    fig.colorbar(c,ax=ax2)
    fig.colorbar(c,ax=ax3)

    fig.suptitle('Bet-hedge variance: '+str(prefvariance)+', Drift variance: '+str(driftvariance), y=-.05, fontsize=16)

    plt.show()

    #Add in age