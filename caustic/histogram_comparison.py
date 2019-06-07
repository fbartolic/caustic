import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os, random
import sys
import exoplanet as xo
import theano
import corner
import seaborn as sns
import pandas as pd

from data import OGLEData
from models import PointSourcePointLens
from models import FiniteSourcePointLens
from utils import plot_model_and_residuals


random.seed(42)

events = []  # data for each event

data_path = '/home/star/fb90/data/OGLE_ews/2017/'
dirs = []
for directory in os.listdir(data_path):
    dirs.append(directory)

random.shuffle(dirs)
#i = 0
#n_events = 200
#for directory in dirs:
#    if (i < n_events):
#        event = OGLEData(data_path + directory)
#        events.append(event)
#        i = i + 1

event = OGLEData(data_path + 'blg-0627')
events.append(event)
     
# Define output directories
output_dir1 = 'output/' + event.event_name +\
        '/PointSourcePointLens/'

output_dir2 = 'output/' + event.event_name +\
        '/PointSourcePointLensGP/'

with PointSourcePointLens(event, kernel='white_noise', 
    errorbar_rescaling='additive_variance') as model1:
    print("Initializing model.")

with PointSourcePointLens(event, kernel='matern32', 
    errorbar_rescaling='additive_variance') as model2:
    print("Initializing model.")

# Sample the model
with model1:
        trace1 = pm.load_trace(output_dir1 + '/model.trace')

with model2:
        trace2 = pm.load_trace(output_dir2 + '/model.trace')

tE_1 = trace1['tE']
tE_2 = trace2['tE']

print(np.shape(tE_1))

#fig, ax = plt.subplots()
#ax.hist(tE_1, color='C0', label='no GP', normed=True, alpha=0.5, bins=100)
#ax.hist(tE_2, color='C1', label='GP', normed=True, alpha=0.5, bins=100)
#ax.set_xlim(0, 200)

sns.set()

mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 18

fig, ax = plt.subplots()
ax.hist(tE_1, bins=15, normed=True, alpha=0.75, label='White noise model')
ax.hist(tE_2, bins=500, normed=True, alpha=0.75, label='Correlated noise model')
ax.set_xlim(0, 125)
ax.set_xlabel(r'Einstein crossing time $t_E$ [days]')
ax.legend(prop={'size': 16})
ax.set_title('Posterior samples for parameter $t_E$')
plt.savefig(output_dir1 + 'histogram.pdf', bbox_inches='tight')

#ax = sns.violinplot(data)
#sns.violinplot([tE_1, tE_2])
#sns.distplot([tE_1, tE_2])
#sns.distplot(tE_1, norm_hist=True)
#sns.distplot(tE_2, norm_hist=True)

#ax.set_xlim(0, 100)
plt.show()