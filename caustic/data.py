import pandas as pd  
import numpy as np 
from matplotlib import pyplot as plt
from io import StringIO
import re
from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.table import Table

class Data(object):
    """
    Abstract base class for microlensing data from various observatories. 
    Subclasses should overload the :func:`Data.load_data`. The time series
    data is stored as a list of Astropy tables, one for each photometric band.
    """
    def __init__(self, event_dir=""):
        self.tables = []
        self.masks = []
        self.event_name = ""
        self.coordinates = None
        self.units = 'magnitudes' # units have to be the same across all bands

    def __str__(self):
        print(self.tables)

    def __add__(self, other):
        """
        Defines an addition operation between datasets. Given multiple 
        observations of the same event, one can load each dataset seperately 
        and simply add them together.
        """
        result = Data()
        if (self.units != other.units):
            raise ValueError('Make sure that all datasets have the same units\
                before adding them.')
        result.tables = self.tables + other.tables # concatonates tables
    
        if (self.coordinates is not None):
            result.coordinates = self.coordinates
        elif (other.coordinates is not None):
            result.coordinates = self.coordinates

        return result

    def load_data(self, event_dir):
        """
        Loads raw time series data for each survey into Astropy tables,
        stores it in `tables` class atrribute.
        """

    def convert_data_to_fluxes(self):
        """
        If the light curves stored in `tables` attribute are expressed in 
        magnitudes, calling this function will convert them to fluxes.
        
        Raises
        ------
        ValueError
            Data is already in flux units.
        
        """
        if not (self.units=='magnitudes'):
            raise ValueError('Data is already in flux units.')

        for table in self.tables:
            F, F_err = self.magnitudes_to_fluxes(table['mag'], 
                table['mag_err'], zero_point=22.)

            table.rename_column('mag', 'flux')
            table.rename_column('mag_err', 'flux_err')
            table['flux'] = F
            table['flux_err'] = F_err

        self.units = 'fluxes'

    def convert_data_to_magnitudes(self):
        """
        If the light curves stored in `tables` attribute are expressed in 
        fluxes, calling this function will convert them to magnitudes.
        
        Raises
        ------
        ValueError
            Data is already in magnitude units.
        
        """
        if not (self.units=='fluxes'):
            raise ValueError("Data is already in magnitude units.")

        for table in self.tables:
            m, m_err = self.fluxes_to_magnitudes(table['flux'], 
                table['flux_err'], zero_point=22.)

            table.rename_column('flux', 'mag')
            table.rename_column('flux_err', 'mag_err')
            table['mag'] = m
            table['mag_err'] = m_err

        self.units = 'magnitudes'

    def magnitudes_to_fluxes(self, m, sig_m, zero_point=22.):
        """
        Given the mean and the standard deviation of a astronomical magnitude
        which is assumed to be normally distributed, and a reference magnitude,
        this function returns the mean and the standard deviation of a flux, 
        which is log-normally distributed.
        
        Parameters
        ----------
        m : ndarray 
            Array of magnitude values.
        sig_m : ndarray
            Array of standard deviations associated with the magnitude array, 
            assumed to be normally distributed. 
        zero_point : float, optional
            Magnitude at which flux is defined to be 1. (the default is 22.)
        
        Returns
        -------
        tuple
            Tuple of ndarrays (mu_F, sig_F) where mu_F is the mean of the 
            log-normal distributed flux, and sig_F is the corresponding 
            square root variance.
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
        Given the mean and the standard deviation of a measured flux 
        which is assumed to be log-normal distributed, and a reference magnitude,
        this function returns the mean and the standard deviation of an 
        astronomical magnitude, which is normally distributed. This function
        is the inverse of :func:`magnitudes_to_fluxes`.
        
        Parameters
        ----------
        F : ndarray 
            Array of flux values.
        sig_F : ndarray
            Array of standard deviations associated with the flux array, 
            assumed to be log-normal distributed. 
        zero_point : float, optional
            Magnitude at which flux is defined to be 1. (the default is 22.)
        
        Returns
        -------
        tuple
            Tuple of ndarrays (mu_F, sig_F) where mu_F is the mean of the 
            log-normal distributed flux, and sig_F is the corresponding 
            square root variance.
        """
        e = np.exp(1)
        sig_m = 2.5*np.log10(e)*np.sqrt(np.log(sig_F**2/F**2 + 1))
        mu_m = zero_point - 2.5*np.log10(e)*(np.log(F) -\
             0.5*np.log(1 + sig_F**2/F**2)) 

        return mu_m, sig_m

    def get_standardized_data(self):  
        """
        This function standardizes the data, expressed in flux units, to zero 
        median and unit variance, a format which is more suitable for 
        subsequent modeling.
        """
        if not (self.units=='fluxes'):
            self.convert_data_to_fluxes()

        # Subtract the median from the data such that baseline is at approx 
        # zero, rescale the data such that it has unit variance
        stanardized_data = []
        for i, table in enumerate(self.tables):
            mask = self.masks[i]

            table_std = Table()
            table_std.meta = table.meta
            table_std['flux'] = (table['flux'][mask] -\
                np.median(table['flux'][mask]))/np.std(table['flux'][mask])
            table_std['flux_err'] = table['flux_err'][mask]/np.std(table['flux'][mask])
            table_std['HJD'] = table['HJD'][mask] - 2450000
            stanardized_data.append(table_std)

        return stanardized_data

    def plot(self, ax):
        """
        Plots raw data.
        
        Parameters
        ----------
        ax : Matplotlib axes object

        """
        if (self.units=='fluxes'):
            for i, table in enumerate(self.tables):
                mask = self.masks[i]

                # Plot data
                ax.errorbar(table['HJD'][mask] - 2450000, table['flux'][mask], 
                    table['flux_err'][mask], fmt='.', color='C' + str(i), 
                    label=table.meta['band'], ecolor='C' + str(i))
                ax.set_ylabel('flux')

                # Plot outliers
                ax.errorbar(table['HJD'][~mask] - 2450000,
                    table['flux'][~mask], 
                    table['flux_err'][~mask], fmt='.', color='C' + str(i),
                    ecolor='C' + str(i), alpha=0.15)
        else:
            for i, table in enumerate(self.tables):
                ax.errorbar(table['HJD'] - 2450000, table['mag'], 
                    table['mag_err'], fmt='.', color='C' + str(i), 
                    label=table.meta['band'], ecolor='C' + str(i))
                ax.set_ylabel('mag')
                ax.invert_yaxis()

        ax.set_xlabel('HJD - 2450000')
        ax.set_title(self.event_name)
        ax.grid(True)
        ax.legend(prop={'size': 16})

    def plot_standardized_data(self, ax):
        """
        Plots data in standardized modeling format.
        
        Parameters
        ----------
        ax : Matplotlib axes object
        """
        std_tables = self.get_standardized_data()

        # Plot masked data
        for i, table in enumerate(std_tables):
            ax.errorbar(table['HJD'], table['flux'], 
                table['flux_err'], fmt='.', color='C' + str(i), 
                label=table.meta['band'], ecolor='C' + str(i))

        ax.grid(True)
        ax.set_title(self.event_name)
        ax.set_xlabel('HJD - 2450000')
        ax.set_ylabel('Flux (rescaled)')
        ax.legend(prop={'size': 16})

    def remove_worst_outliers(self, window_size=7, mad_cutoff=5):
        tables = self.tables
        for i, table in enumerate(tables):
            series = pd.Series(table['flux']) 
            mad = lambda x: 1.4826*np.median(np.abs(x - np.median(x)))
            rolling_mad = np.array(series.rolling(window_size, 
                center=True).apply(mad, raw=True))
            rolling_mad[-window_size//2:] = rolling_mad[-window_size//2]
            rolling_mad[:window_size//2] = rolling_mad[window_size//2]
            rolling_median = np.array(series.rolling(window_size, center=True).median())
            rolling_median[-window_size//2:] = rolling_median[-window_size//2]
            rolling_median[:window_size//2] = rolling_median[window_size//2]
            
            array = np.abs((np.array(table['flux']) - rolling_median)/rolling_mad)
            mask = array > 5
            
            # Update masks
            self.masks[i] = ~mask

    def reset_masks(self):
        for i in range(len(self.masks)):
            self.masks[i] = np.ones(len(self.masks[i]), dtype=bool)
            
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
        self.load_data(event_dir)

    def load_data(self, event_dir):
        """Returns a table with raw data."""
        t = Table.read(event_dir + '/phot.dat', format='ascii', 
            names=('HJD', 'mag', 'mag_err', 'col4', 'col5'))
        t.keep_columns(('HJD', 'mag', 'mag_err'))
        t.meta = {'band':'OGLE I', 'observatory':'OGLE'}
        self.tables.append(t)
        self.masks = [np.ones(len(t['HJD']), dtype=bool)]

class MOAData(Data):
    """Subclass of data class for dealing with OGLE data."""
    def __init__(self, event_path, index_path):
        super(MOAData, self).__init__(event_path)
        self.units = 'fluxes'

        # Grabbing the event name is anything but trivial
        with open(event_path) as f:
            contents = f.readlines()
            processed = ''
            for i in range(len(contents)):
                processed += re.sub("\s+", ",", contents[i].strip()) + '\n' 
            processed = StringIO(processed)
            table = pd.read_csv(processed, sep=',', header=None, skiprows=2,
                nrows=5)

        event_code = table[2].loc[0] + '-' + table[2].loc[1] + '-' +\
             table[2].loc[2] + '-' + table[2].loc[3]

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
            t = Table.read(processed, format='ascii')
            t.keep_columns(['col1', 'col2', 'col3'])
            t.rename_column('col1', 'HJD')
            t.rename_column('col2', 'mag')
            t.rename_column('col3', 'mag_err')
            t.meta = {'band':'MOA I', 'observatory':'MOA'}

            # Remove the random rows with zero time and negative time
            t = t[t['HJD'] > 0]

        self.tables.append(t)

class LCOData(Data):
    """Subclass of data class for dealing with ROME_REA data."""
    def __init__(self, event_path):
        super(LCOData, self).__init__(event_path)

    def load_data(self, event_path):
        """Returns dataframe with raw data."""
        table = pd.read_csv(event_path, sep=' ',  usecols=(1, 13, 14), 
            names=['HJD - 2450000', 'mag', 'mag_err'])

        table['HJD - 2450000'] = table['HJD - 2450000'] - 2450000 
        return table

class KMTData(Data):
    """Subclass of data class for dealing with OGLE data."""
    def __init__(self, event_dir):
        super(KMTData, self).__init__(event_dir)
        self.load_data(event_dir)
        self.units = 'fluxes'

    def load_data(self, event_dir):
        """Returns a table with raw data."""
        t1 = Table.read(event_dir + '/KMTA01_I.diapl', format='ascii')
        t1['col1'] += 2450000
        t1.keep_columns(('col1', 'col2', 'col3'))
        t1.rename_column('col1', 'HJD')
        t1.rename_column('col2', 'flux')
        t1.rename_column('col3', 'flux_err')
        t1.meta = {'band':'KMTA_I', 'observatory':'KMTNet'}

        t2 = Table.read(event_dir + '/KMTC01_I.diapl', format='ascii')
        t2['col1'] += 2450000
        t2.keep_columns(('col1', 'col2', 'col3'))
        t2.rename_column('col1', 'HJD')
        t2.rename_column('col2', 'flux')
        t2.rename_column('col3', 'flux_err')
        t2.meta = {'band':'KMTC_I', 'observatory':'KMTNet'}

        t3 = Table.read(event_dir + '/KMTS01_I.diapl', format='ascii')
        t3['col1'] += 2450000
        t3.keep_columns(('col1', 'col2', 'col3'))
        t3.rename_column('col1', 'HJD')
        t3.rename_column('col2', 'flux')
        t3.rename_column('col3', 'flux_err')
        t3.meta = {'band':'KMTS_I', 'observatory':'KMTNet'}

        self.tables = [t1, t2, t3]
        self.masks = [np.ones(len(t1['HJD']), dtype=bool), 
            np.ones(len(t2['HJD']), dtype=bool),
            np.ones(len(t3['HJD']), dtype=bool)]