import simpledrift as sd
import matrixmaker as mm
import numpy as np
# import imp
# import matplotlib.pyplot as plt
import os
# import math
import xarray
from datetime import date

i = os.getenv('SLURM_ARRAY_TASK_ID')

mm.frequency_phaseplane(i=i)