import matplotlib.pyplot as plt
import numpy as np
import simpledrift as sd

#[finalpop]=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean[q], prefvariance[q], envimean, envivariance, driftvariance[q], gain, per, deathrate, birthrate, matureage, percentbh)

#sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance, envimean, envivariance, driftvariance, gain, per,maxsurvivalrate,birthrate,matureage, percentbh)

def heatmap(finalpop):
    finalpoplog=np.log(finalpop)
    plt.pcolor(finalpoplog, cmap='hot')
    plt.colorbar()
    plt.show()