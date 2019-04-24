import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano

from data import KMTData
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensMarginalized
from simulation_based_calibration import SBC

random.seed(42)
kmt_dir = '/home/star/fb90/data/KMT/kmtnet/2017/2017/KB170053'
event = KMTData(kmt_dir)
event.event_name = 'KMTKB170053'

# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLens'

# Create output directory
if not os.path.exists(output_dir1):
    os.makedirs(output_dir1)


def my_model(y=None):
    with PointSourcePointLens(event, 
        errorbar_rescaling='additive_variance') as model_instance:
        print('test')
   
    return model_instance


sbc = SBC(my_model, 'y',
        num_simulations=1000,
        sample_kwargs={'draws': 25, 'tune': 50})

sbc.run_simulations()
