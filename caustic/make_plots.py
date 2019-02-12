from data import OGLEData
from utils import plot_model_and_residuals
from matplotlib import pyplot as plt
from models import PointSourcePointLens
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
        output_dir = 'output/' + event.event_name + '/PointSourcePointLens/' 
        model_standard = PointSourcePointLens(event, use_joint_prior=False)
        
        with model_standard:
            trace_standard = pm.load_trace(output_dir + 'model.trace') 

        # Plot traceplots
        #plot_traceplots(trace_standard, 15, output_dir)

        # Plot model
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]},
                figsize=(25, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)
        plot_model_and_residuals(ax, event, model_standard, trace_standard, 0,
            len(event.df['HJD - 2450000']) - 1)
        plt.savefig(output_dir + 'model.png')

        # Trace in dataframe format
        df = pm.trace_to_dataframe(trace_standard)
        df.to_csv(output_dir + 'data.csv')