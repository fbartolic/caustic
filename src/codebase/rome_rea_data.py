from Data import OGLEData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from io import StringIO

fig, ax = plt.subplots(figsize=(20, 5))

with open('../../../../data/ROME_REA/' + 'lc_00299.583_00302.981_t') as f:
    contents = f.readlines()
    processed = ''
    for i in range(len(contents)):
        processed += '\n' + contents[i][164:] 
    processed = StringIO(processed)
    df = pd.read_csv(processed, sep=' ',  usecols=(1, 13, 14), 
            names=['HJD - 2450000', 'I_mag', 'I_mag_err'])
    print(df.head())

ax.plot(df['HJD - 2450000'].values, df['I_mag'].values, 'k.')
plt.show()
    
