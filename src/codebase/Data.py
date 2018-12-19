import pandas as pd  
import numpy as np 
from matplotlib import pyplot as plt

class Data(object):
    """
    Abstract base class for microlensing data from various observatories. 
    Subclasses should overload the :func:`Data.load_data`.
    """
    def __init__(self, event_dir=""):
        if (event_dir[-1]=='/'):
            event_dir = event_dir[:-1]
        self.df = self.load_data(event_dir)
        self.event_name = ""
        self.RA = ""
        self.Dec = ""
        self.filters = ['']
        self.units = 'magnitudes'
        self.observatory = ''

    def __add__(self, another_dataset):
        """"""
    def __str__(self):
        return "Event name: {}\nRA: {}\nDec: {}\nFilters: {}".format(\
            self.event_name, self.RA, self.Dec, self.filters)
    
    def load_data(self, event_dir):
        """Returns dataframe with raw data."""

    def convert_data_to_fluxes(self):
        """
        If the lightcurve is stored in magnitudes, calling this function will
        convert it to fluxes.
        
        Raises
        ------
        ValueError
            Data is already in flux units.
        
        """

        if not (self.units=='magnitudes'):
            #raise ValueError("Data is already in flux units.")
            return None

        F, F_err = self.magnitudes_to_fluxes(self.df['I_mag'], 
           self.df['I_mag_err'], zero_point=22.)

        self.df.rename(columns={'I_mag': 'I_flux',
            'I_mag_err':'I_flux_err'}, inplace=True)
        self.df['I_flux'] = F
        self.df['I_flux_err'] = F_err
        self.units = 'fluxes'

    def convert_data_to_magnitudes(self):
        """
        If the lightcurve is stored in flux units, calling this function will
        convert it to magnitudes.
        
        Raises
        ------
        ValueError
            Data is already in magnitude units.
        
        """
        if not (self.units=='fluxes'):
            #raise ValueError("Data is already in magnitude units.")
            return None

        """"Returns a df with the data in expressed in magnitudes rather than
        fluxes."""
        m, m_err = self.fluxes_to_magnitudes(self.df['I_flux'], 
           self.df['I_flux_err'])

        self.df.rename(columns={'I_flux': 'I_mag',
            'I_flux_err':'I_mag_err'}, inplace=True)
        self.df['I_mag'] = m
        self.df['I_mag_err'] = m_err
        self.units = 'magnitudes'

    def magnitudes_to_fluxes(self, m, sig_m, zero_point=22.):
        """
        Given the mean and the standard deviation of a magnitude, 
        assumed to be normally distributed, and a reference magnitude m0, this 
        function returns the mean and the standard deviation of a Flux, 
        which is log-normally distributed.
        """

        # Calculate the mean and std. deviation for lnF which is assumed to be 
        # normally distributed 
        e = np.exp(1)
        mu_lnF = (zero_point - m)/(2.5*np.log10(e)) 
        sig_lnF = sig_m/(2.5*np.log10(e))

        # If lnF is normally distributed, F is log-normaly distributed with a mean
        # and root-variance given by
        mu_F = np.exp(mu_lnF + 0.5*sig_lnF**2)
        sig_F = np.sqrt((np.exp(sig_lnF**2) - 1)*np.exp(2*mu_lnF + sig_lnF**2))

        return mu_F, sig_F    

    def fluxes_to_magnitudes(self, F, sig_F, zero_point=22):
        """
        Does the same thing as `magnitudes_to_fluxes` except in reverse.
        """
        e = np.exp(1)
        sig_m = 2.5*np.log10(e)*np.sqrt(np.log(sig_F**2/F**2 + 1))
        mu_m = zero_point - 2.5*np.log10(e)*(np.log(F) -\
             0.5*np.log(1 + sig_F**2/F**2)) 

        return mu_m, sig_m

    def get_standardized_data(self):  
        """
        If the lightcurve is expressed in flux units, this function standardizes
        the data to zero median and unit variance, a format which is suitable
        for modeling.
        """
        if not (self.units=='fluxes'):
            raise ValueError("Make sure that the units are fluxes instead of \
                magnitudes.")

        # Subtract the median from the data such that baseline is at approx 
        # zero, rescale the data such that it has unit variance
        df = self.df.copy()
        df['I_flux'] = (df['I_flux'] - df['I_flux'].median())/df['I_flux'].std()
        df['I_flux_err'] = df['I_flux_err']/df['I_flux'].std()
        return df

    def plot(self, ax):
        """
        Plots data.
        
        Parameters
        ----------
        ax : Matplotlib axes object
        
        """

        if (self.units=='fluxes'):
            ax.errorbar(self.df['HJD - 2450000'], self.df['I_flux'], 
                self.df['I_flux_err'], fmt='.', color='black', label='Data', 
                ecolor='#686868')
            ax.grid(True)
            ax.set_xlabel(self.df.columns[0])
            ax.set_ylabel(self.df.columns[1])
        else:
            ax.errorbar(self.df['HJD - 2450000'], self.df['I_mag'], 
                self.df['I_mag_err'], fmt='.', color='black', label='Data', 
                ecolor='#686868')
            plt.gca().invert_yaxis()
            ax.grid(True)
            ax.set_xlabel(self.df.columns[0])
            ax.set_ylabel(self.df.columns[1])
    
class OGLEData(Data):
    """Subclass of data class for dealing with OGLE data."""
    def __init__(self, event_dir):
        super(OGLEData, self).__init__(event_dir)
        with open(event_dir + '/params.dat') as f:
            lines = f.readlines() 
            self.event_name = lines[0]
            self.RA = lines[4][15:]
            self.Dec = lines[5][15:]
        self.filters = ['OGLE I band']
        self.observatory = 'OGLE'

    def load_data(self, event_dir):
        """Returns dataframe with raw data."""
        df = pd.read_csv(event_dir + '/phot.dat', sep=' ',  usecols=(0,1,2), 
            names=['HJD - 2450000', 'I_mag', 'I_mag_err'])
        df['HJD - 2450000'] = df['HJD - 2450000'] - 2450000 
        return df

class MOAData(Data):
    """Subclass of data class for dealing with OGLE data."""
    def __init__(self, event_path):
        super(MOAData, self).__init__(event_path)
        self.observatory = 'MOA'

    def load_data(self, event_path):
        """Returns dataframe with raw data."""
        with open(event_path) as f:
            lines = f.readlines() 
            contents = f.readlines()
            processed = ''
            for i in range(len(contents)):
                processed += re.sub("\s+", ",", contents[i].strip()) + '\n' 
            processed = StringIO(processed)
            df = pd.read_csv(processed, sep=',', header=None, skiprows=10)
        return df

class LCOData(Data):
    pass

class KMTData(Data):
    pass