import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt
import time
import simpledrift as sd


def heatmap():
[finalpop]=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean[q], prefvariance[q], envimean, envivariance, driftvariance[q], gain, per,deathrate,birthrate,matureage, percentbh)
    print(finalpop)
    # plt.imshow(finalpop, cmap='hot', interpolation='nearest')
    # plt.show()