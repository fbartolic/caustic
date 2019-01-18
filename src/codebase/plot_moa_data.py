from Data import MOAData
from Data import OGLEData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from io import StringIO
import re

from urllib.parse import urlencode
from urllib.request import urlretrieve
from astropy import units as u

fig, ax = plt.subplots(figsize=(20, 5))

#event_dir ='/home/star/fb90/data/MOA/' + 'KKB180317I.new'
#index_path ='/home/star/fb90/data/MOA/index.txt'
#
#event = MOAData(event_dir, index_path)
#event.plot(ax)
#print(np.sort(event.df['HJD - 2450000'].values))
#plt.show()

event_dir ='/home/star/fb90/data/OGLE_ews/2017/' + 'blg-0045'
event = OGLEData(event_dir)
print(event.coordinates)

# tell the SDSS service how big of a cutout we want
im_size = 12*u.arcmin # get a 12 arcmin square
im_pixels = 1024
cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
query_string = urlencode(dict(ra=event.coordinates.ra.deg,
                              dec=event.coordinates.dec.deg,
                              width=im_pixels, height=im_pixels,
                              scale=im_size.to(u.arcsec).value/im_pixels))
url = cutoutbaseurl + '?' + query_string

# this downloads the image to your disk
urlretrieve(url, 'HCG7_SDSS_cutout.jpg')