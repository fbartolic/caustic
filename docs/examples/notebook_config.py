get_ipython().magic("matplotlib inline")
get_ipython().magic('config InlineBackend.figure_format = "retina"')

# Hide deprecation warnings from Theano
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Hide Theano compilelock warnings
import logging

logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)

import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"