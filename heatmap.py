import matplotlib.pyplot as plt
import numpy as np

def heatmap:
    [finalpop]=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean[q], prefvariance[q], envimean, envivariance, driftvariance[q], gain, per, deathrate, birthrate, matureage, percentbh)

    plt.imshow(finalpop, cmap='hot', interpolation='nearest')
    plt.show()