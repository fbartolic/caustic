import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import os, random
import sys
import exoplanet as xo
import theano.tensor as T

from data import KMTData
from models import PointSourcePointLens
from models import PointSourcePointLensMatern32
from models import PointSourcePointLensMarginalized
from models import OutlierRemovalModel
from utils import plot_map_model_and_residuals

def test_PointSourcePointLens():
    # Load example microlensing event

    # Generate mock dataset

    # Fit mock dataset



    pass