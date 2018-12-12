class Data(object):
    """The abstract base class for microlensing data from various telescopes.
    
    Subclasses should overload the :func:`Term.get_raw_data`."""

    observatory_names = tuple()

    def __init__(self, *args):
        self.F = F
        self.t = t
        self.err_F = err_F
        self.filter = None
    
    def __add__(self, b):
        return DataSum(self, b)

    def plot_raw_data(self, ax):
        """"Plots raw data in whatever format is default for the observatory."""

    def plot_standardized_data(self, ax):
        """Plots data in standard modeling format."""
