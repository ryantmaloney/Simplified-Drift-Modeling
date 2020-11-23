import matplotlib.pyplot as plt
import numpy as np
import simpledrift as sd

[finalpop]=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean[q], prefvariance[q], envimean, envivariance, driftvariance[q], gain, per, deathrate, birthrate, matureage, percentbh)

def heatmap():
    plt.imshow(finalpop, cmap='hot', interpolation='nearest')
    plt.show()