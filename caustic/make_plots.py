import numpy as np
from data import OGLEData
from utils import plot_model_and_residuals
from matplotlib import pyplot as plt
from models import PointSourcePointLensWhiteNoise1
from models import PointSourcePointLensMatern32
import pymc3 as pm
import os 

events = [] # event names
lightcurves = [] # data for each event
data_path = '/home/star/fb90/data/OGLE_ews/2017/'

# Iterate over events, load data, and make plots 
for entry in os.scandir('output'):
    if entry.is_dir():
        data_dir = data_path + 'blg-' + entry.name[-4:]
        event = OGLEData(data_dir)
        print(event.event_name)

        # Load traces
        output_dir = 'output/' + event.event_name + '/PointSourcePointLensMatern32/' 

        # Plot non-GP models
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
            figsize=(25, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)

        plot_model_and_residuals(ax, event, 
        PointSourcePointLensMatern32(event), output_dir + 'model.trace')
        plt.savefig(output_dir + 'model.png')

        plt.show()

