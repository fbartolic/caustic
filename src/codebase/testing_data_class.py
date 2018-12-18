from Data import OGLEData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from io import StringIO
import re

fig, ax = plt.subplots(figsize=(20, 5))

event_dir ='../../../../data/MOA/' + 'phot-gb1-R-7-4500.dat'

with open(event_dir) as f:
    contents = f.readlines()
    processed = ''
    for i in range(len(contents)):
        processed += re.sub("\s+", ",", contents[i].strip()) + '\n' 
    processed = StringIO(processed)
    df = pd.read_csv(processed, sep=',', header=None, skiprows=10)
    print(df.head())

print(np.argmin(df[0].values))

ax.plot(df[0].values, df[1].values, 'k.')
plt.show()