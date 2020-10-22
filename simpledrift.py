import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt

def driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance, envimean, envivariance, driftvariance, gain, per):
    x=np.linspace(-1,1,numberofbins)
    pref=np.zeros((numberofbins,numberofdays))
    pref[:,0]=sci.norm.pdf(x,prefmean,prefvariance) # A gaussian of preference with center around 0
    pref[:,0]=pref[:,0]/np.sum(pref[:,0])*flynum
    #envi=gain*np.sin(x*2*np.pi/per+182*2*np.pi)+envimean
    # envi=gain*np.sin(x*2*np.pi/per+2*np.pi)+envimean
    # x=np.linspace(0, numberofdays, numberofbins)
    # envi=np.sin(x)+envimean
    # print(np.max(envi))
    envi=sci.norm.pdf(x,envimean,envivariance) # A gaussian of environment with center around 0
    # print(np.max(envi))
    envi=envi/(np.max(envi))*.95
    for t in range(1,numberofdays):
        for b in range(numberofbins):
            pref[:,t]+=pref[b,t-1]*sci.norm.pdf(x,x[b],driftvariance)/np.sum(sci.norm.pdf(x,x[b],driftvariance))
            # print(np.sum(sci.norm.pdf(x,x[b],driftvariance)))
        # plt.plot(pref[:,t])
        # plt.show()
        pref[:,t]=np.multiply(pref[:,t], envi) # Multiplying the preference to the environment
        # print(pref[:,t])
        pref[:,t]=pref[:,t]+pref[:,0]*0.2/flynum*np.sum(pref[:,t])

    plt.pcolormesh(pref)
    plt.colorbar()