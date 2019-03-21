import numpy as np
from matplotlib import pyplot as plt
import os 
import sys
import pandas as pd


df = pd.DataFrame(columns=['name', 'DeltaF_mean', 'DeltaF_sd', 'Fbase_mean', 'Fbase_sd'])

# Iterate over events, load data, and make plots 
for entry in os.scandir('output'):
    if entry.is_dir():
        data = pd.read_csv('output/' + entry.name + '/PointSourcePointLensWN3/sampling_stats.csv')
        df = df.append({'name':entry.name,
                       'DeltaF_mean':data['mean'][1],
                       'DeltaF_sd':data['sd'][1],
                       'Fbase_mean':data['mean'][0],
                       'Fbase_sd':data['sd'][0]}, ignore_index=True)


ax = df['DeltaF_mean'].hist()
ax.set_xlabel('mean Delta_F')

plt.show()