import matplotlib.pyplot as plt
import numpy as np
import simpledrift as sd

#[finalpop]=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean[q], prefvariance[q], envimean, envivariance, driftvariance[q], gain, per, deathrate, birthrate, matureage, percentbh)

#sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance, envimean, envivariance, driftvariance, gain, per,maxsurvivalrate,birthrate,matureage, percentbh)

def hello(hi):
    sigh=hi+3
    return sigh

def bye(sigh):
    hello(hi)


# def heatmap(matrix):
#     matrixlog=np.log(matrix)
#     plt.pcolor(matrixlog, cmap='hot')
#     plt.colorbar()
#     plt.show()