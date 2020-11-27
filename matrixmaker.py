# make the matrix
# call simpledrift to fill in the values for different drift and bh
# pass on to heatmap
# pay attention to axes so they give the correct values
import numpy as np
import matplotlib.pyplot as plt
import simpledrift as sd


def matrixmaker(bhlower, bhupper, driftlower, driftupper):

    flynum=1
    numberofbins=100
    numberofdays=50
    prefmean=0
    envimean=0
    envivariance=.25
    gain=.4
    per=20
    maxsurvivalrate=1
    birthrate=40
    matureage=10
    percentbh=0.01
    intervals=10
    matrix=np.zeros((intervals,intervals))
    prefvariance=np.linspace(bhlower, bhupper, intervals)
    driftvariance=np.linspace(driftlower, driftupper, intervals)

    for x in range(intervals):
        for y in range(intervals):
            finalpop=sd.driftmodeling(flynum, numberofbins, numberofdays, prefmean, [prefvariance[x]], envimean, envivariance, [driftvariance[y]], gain, per,maxsurvivalrate,birthrate,matureage, percentbh)
            matrix[x,y]=finalpop
    print(matrix)

    matrixlog=np.log(matrix)

    fig,ax=plt.subplots()
    c=ax.pcolor(matrixlog, cmap='hot')
    ax.set_xlabel('Bet Hedging')
    ax.set_ylabel('Drift')
    fig.colorbar(c)
    ax.set_title('Log of Final Population')
    plt.show()

    return matrix
        
        