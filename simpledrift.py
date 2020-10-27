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
    for t in range(1,numberofdays):
        envi[:,t]=sci.norm.pdf(x,(envimean+gain*np.sin(t*np.pi*2/per)),envivariance)
        envi[:,t]=envi[:,t]/np.max(envi[:,t])*.95
        for b in range(numberofbins):
            for a in range(maxage):
                pref[:,t,a]+=pref[b,t-1,a]*sci.norm.pdf(x,x[b],driftvariance)/np.sum(sci.norm.pdf(x,x[b],driftvariance))
            # print(np.sum(sci.norm.pdf(x,x[b],driftvariance)))
        # plt.plot(pref[:,t])
        # plt.show()
                pref[:,t,a]=np.multiply(pref[:,t,a], envi[:,t]) # Multiplying the preference to the environment
        # print(pref[:,t])
                if a > 9:
                    pref[:,t,0]=pref[:,t,0]+pref[:,0,a]*birthrate/flynum*np.sum(pref[:,t-1,a])

    plt.pcolormesh(np.sum(pref,axis=2))
    plt.colorbar()
    plt.title('Fly Preference (color is num flies each day')
    plt.ylabel('Preference')
    plt.xlabel('Day')
    plt.show()
    plt.pcolormesh(envi)
    plt.colorbar()
    plt.title('Environment (color is fraction of flies of given pref survive)')
    plt.ylabel('Preference')
    plt.xlabel('Day')
    plt.show()

    #Add in age