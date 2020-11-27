# make the matrix
# call simpledrift to fill in the values for different drift and bh
# pass on to heatmap
# pay attention to axes so they give the correct values
import numpy as np
import matplotlib.pyplot as plt
import simpledrift as sd

def matrixmaker(bhlower, bhupper, driftlower, driftupper):
    matrix=np.zeros((len(range(driftlower, driftupper)), len(range(bhlower, bhupper))))
    for x in range(bhlower, bhupper):
        for y in range(driftlower, driftupper):
            sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance[x], envimean, envivariance, driftvariance[y], gain, per, maxsurvivalrate, birthrate, matureage, percentbh)
            finalpop=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean, prefvariance, envimean, envivariance, driftvariance, gain, per,maxsurvivalrate,birthrate,matureage, percentbh)
            matrix[x,y]=finalpop
    print(matrix)
        
        