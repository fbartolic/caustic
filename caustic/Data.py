import pandas as pd  
import numpy as np 
from matplotlib import pyplot as plt
from io import StringIO
import re
from astropy.coordinates import SkyCoord
from astropy import units as u

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
        self.coordinates = None
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
            self.convert_data_to_fluxes()

        # Subtract the median from the data such that baseline is at approx 
        # zero, rescale the data such that it has unit variance
        df_std = self.df.copy()
        df_std['I_flux'] = (self.df['I_flux'] -\
             self.df['I_flux'].median())/np.std(self.df['I_flux'].values)
        df_std['I_flux_err'] = self.df['I_flux_err']/np.std(self.df['I_flux'].values)
        return df_std

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
        else:
            ax.errorbar(self.df['HJD - 2450000'], self.df['I_mag'], 
                self.df['I_mag_err'], fmt='.', color='black', label='Data', 
                ecolor='#686868')
            ax.invert_yaxis()

        ax.set_xlabel(self.df.columns[0])
        ax.set_ylabel(self.df.columns[1])
        ax.set_title(self.event_name)
        ax.grid(True)

    def plot_standardized_data(self, ax, mask=None):
        """
        Plots data in standardized modeling format.
        
        Parameters
        ----------
        ax : Matplotlib axes object
        
        mask : Integer array
        """
        df = self.get_standardized_data()
        t = df['HJD - 2450000'].values[mask]
        F = df['I_flux'].values[mask]
        F_err = df['I_flux_err'].values[mask]

        ax.errorbar(t, F, F_err, fmt='.', color='black', label='Data', 
            ecolor='#686868')
        ax.grid(True)
        ax.set_title(self.event_name)
        ax.set_xlabel('Flux (rescaled)')
        ax.set_ylabel(df.columns[1])
            
class OGLEData(Data):
    """Subclass of data class for dealing with OGLE data."""
    def __init__(self, event_dir):
        super(OGLEData, self).__init__(event_dir)
        with open(event_dir + '/params.dat') as f:
            lines = f.readlines() 
            self.event_name = lines[0][:-1]
            RA = lines[4][15:-1]
            Dec = lines[5][15:-1]
            self.coordinates = SkyCoord(RA, Dec, 
                unit=(u.hourangle, u.deg, u.arcminute))
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
    def __init__(self, event_path, index_path):
        super(MOAData, self).__init__(event_path)
        self.observatory = 'MOA'
        self.units = 'fluxes'

        # Grabbing the event name is anything but trivial
        with open(event_path) as f:
            contents = f.readlines()
            processed = ''
            for i in range(len(contents)):
                processed += re.sub("\s+", ",", contents[i].strip()) + '\n' 
            processed = StringIO(processed)
            df = pd.read_csv(processed, sep=',', header=None, skiprows=2, nrows=5)

        event_code = df[2].loc[0] + '-' + df[2].loc[1] + '-' + df[2].loc[2] +\
                '-' + df[2].loc[3]

        # Load index file, find real name of the event
        index = pd.read_csv(index_path, sep=' ', header=None,
                usecols=(0, 1,), names=['Name', 'code'])
            
        key_value = index[index['code'].str.match(event_code)].iloc[0]
        self.event_name = 'MOA-' + key_value.iloc[0]

    def load_data(self, event_path):
        """Returns dataframe with raw data."""
        # It's not sure that time for MOA data is in HJD
        with open(event_path) as f:
            contents = f.readlines()
            processed = ''
            for i in range(len(contents)):
                processed += re.sub("\s+", ",", contents[i].strip()) + '\n' 
            processed = StringIO(processed)
            df = pd.read_csv(processed, sep=',', header=None, skiprows=10,
                usecols=(0, 1, 2), 
                names=['HJD - 2450000', 'I_flux', 'I_flux_err'])

            df['HJD - 2450000'] = df['HJD - 2450000'] - 2450000 

            # Remove the random rows with zero time and negative time
            df = df[df['HJD - 2450000'] > 0]

        return df

class LCOData(Data):
    """Subclass of data class for dealing with ROME_REA data."""
    def __init__(self, event_path):
        super(LCOData, self).__init__(event_path)
        self.observatory = 'LCO ROME/REA'

    def load_data(self, event_path):
        """Returns dataframe with raw data."""
        df = pd.read_csv(event_path, sep=' ',  usecols=(1, 13, 14), 
            names=['HJD - 2450000', 'I_mag', 'I_mag_err'])

        df['HJD - 2450000'] = df['HJD - 2450000'] - 2450000 
        return df


class KMTData(Data):
    pass