import pandas as pd  
import numpy as np 
from matplotlib import pyplot as plt
from io import StringIO
import re
import os
import copy


from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.table import Table


class Data:
    """
    Base class for microlensing data from various observatories. 

    Subclasses should overload the :func:`Data.__load_data` method. The time 
    series data is stored as a list of Astropy tables, one for each photometric
    filter. Available subclasses for specific data sources are ``OGLEData``,
    ``MOAData``, ``KMTData``, ``NASAExoArchiveData``.

    Example usage:

    .. code-block:: python

        event = caustic.data.OGLEData("path_to_dir")
        coords = event.coordinates  # Get coordinates of the event
        name = event.event_name   # Get event name(s)
        event.units = "fluxes"  # Change units from magitudes to fluxes
        event.units = "magnitudes"  # Change units from fluxes to magnitudes


    Parameters
    ----------
    event_dir : str
        Path to the directory containing microlensing data.
    """
    def __init__(
        self, 
        event_dir=None
    ):
        self.__tables = []
        self.__event_name = ""
        self.__coordinates = None
        self.__units = 'magnitudes' # units have to be the same across bands

    def __str__(self):
        print(self.__tables)

    def __add__(self, other):
        """
        Defines an addition operation between datasets. Given multiple 
        observations of the same event, one can load each dataset seperately 
        and add them together.
        """
        result = Data()

        self.units = 'fluxes'
        other.units = 'fluxes'
        result.units = 'fluxes'

        result.light_curves = self.light_curves + other.light_curves
    
        if (self.event_coordinates is not None and \
            other.event_coordinates is not None):
            if self.event_coordinates==other.event_coordinates:
                result.coordinates = self.__coordinates
            else:
                raise ValueError("Coordinates of the two events need to match.")
        elif (self.event_coordinates is not None):
            result.event_coordinates = self.event_coordinates
        elif (other.event_coordinates is not None):
            result.event_coordinates = other.event_coordinates

        result.event_name = self.event_name + other.event_name

        return result

    def __load_data(self, event_dir):
        """
        Loads raw time series data for each survey into Astropy tables,
        stores it in `tables` class atrribute.
        """

    def __convert_data_to_fluxes(self):
        """
        If the light curves stored in the ``__tables`` attribute are expressed in 
        magnitudes, calling this function will convert them to fluxes.
        """
        if self.__units=='fluxes':
            pass
        
        else:
            for table in self.__tables:
                F, F_err = self.__magnitudes_to_fluxes(table['mag'], 
                    table['mag_err'])

                table.rename_column('mag', 'flux')
                table.rename_column('mag_err', 'flux_err')
                table['flux'] = F
                table['flux_err'] = F_err

            self.__units = 'fluxes'

    def __convert_data_to_magnitudes(self):
        """
        If the light curves stored in the ``__tables`` attribute are expressed in 
        fluxes, calling this function will convert them to magnitudes.
        """
        if self.__units=='magnitudes':
            pass

        else:
            for table in self.__tables:
                m, m_err = self.__fluxes_to_magnitudes(table['flux'], 
                    table['flux_err'])

                table.rename_column('flux', 'mag')
                table.rename_column('flux_err', 'mag_err')
                table['mag'] = m
                table['mag_err'] = m_err

            self.__units = 'magnitudes'

    def __magnitudes_to_fluxes(self, m, sig_m, zero_point=22.):
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
            Tuple of ndarrays ``(mu_F, sig_F)`` where ``mu_F`` is the mean of the 
            log-normal distributed flux, and ``sig_F`` is the corresponding 
            square root variance.
        """
        # Truncate errorbars greater than 1 mag for numerical stability reasons
        sig_m[sig_m > 1] = 1

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

    def __fluxes_to_magnitudes(self, F, sig_F, zero_point=22.):
        """
        Given the mean and the standard deviation of a measured flux 
        which is assumed to be log-normal distributed, and a reference magnitude,
        this function returns the mean and the standard deviation of an 
        astronomical magnitude, which is normally distributed. This function
        is the inverse of :func:`__magnitudes_to_fluxes`.
        
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
            Tuple of ndarrays ``(mu_F, sig_F)`` where mu_F is the mean of the 
            log-normal distributed flux, and ``sig_F`` is the corresponding 
            square root variance.
        """
        e = np.exp(1)
        sig_m = 2.5*np.log10(e)*np.sqrt(np.log(sig_F**2/F**2 + 1))
        mu_m = zero_point - 2.5*np.log10(e)*(np.log(F) -\
             0.5*np.log(1 + sig_F**2/F**2)) 

        return mu_m, sig_m

    def get_standardized_data(self):  
        """
        This function returns data tables in a standardized format, expressed in 
        flux units rescaled to zero  median and unit variance, a format which 
        is more suitable for subsequent modeling. The conversion from fluxes
        to magnitudes defines a flux of 1 to correspond to magnitude 22.
        """
        tmp_data = copy.deepcopy(self)
        if not (tmp_data.units=='fluxes'):
            tmp_data.units = 'fluxes'

        # Subtract the median from the data such that baseline is at approx 
        # zero, rescale the data such that it has unit variance
        standardized_data = []
        for i, table in enumerate(tmp_data.light_curves):
            mask = table['mask']
            table_std = Table()
            table_std.meta = table.meta
            table_std['HJD'] = table['HJD'][mask] - 2450000
            table_std['flux'] = (table['flux'][mask] -\
                np.median(table['flux'][mask]))/np.std(table['flux'][mask])
            table_std['flux_err'] = table['flux_err'][mask]/np.std(table['flux'][mask])
            standardized_data.append(table_std)

        return standardized_data

    def plot(self, ax):
        """
        Plots raw data.
        
        Parameters
        ----------
        ax : Matplotlib axes object
        """
        if (self.__units=='fluxes'):
            unit = 'flux'
        else:
            unit = 'mag'

        for i, table in enumerate(self.__tables):
            mask = table['mask']

            # Plot data
            ax.errorbar(table['HJD'][mask] - 2450000, table[unit][mask], 
                table[unit + '_err'][mask], fmt='o', color='C' + str(i), 
                label=table.meta['observatory'] + ' ' + table.meta['filter'], 
                ecolor='C' + str(i), alpha=0.5)
            ax.set_ylabel(unit)

            # Plot masked data
            ax.errorbar(table['HJD'][~mask] - 2450000,
                table[unit][~mask], 
                table[unit + '_err'][~mask], fmt='o', color='C' + str(i),
                ecolor='C' + str(i), alpha=0.1)

        if (self.__units=='magnitudes'):
            ax.invert_yaxis()

        ax.set_xlabel('HJD - 2450000')
        ax.set_title(self.__event_name)
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
            # Plot data
            ax.errorbar(table['HJD'], table['flux'], 
                table['flux_err'], fmt='o', color='C' + str(i), 
                label=table.meta['observatory'] + ' ' + table.meta['filter'], 
                ecolor='C' + str(i), alpha=0.5)
            ax.set_ylabel('flux')
 
        ax.grid(True)
        ax.set_title(self.__event_name)
        ax.set_xlabel('HJD - 2450000')
        ax.set_ylabel('Flux (rescaled)')
        ax.legend(prop={'size': 16})

    def remove_worst_outliers(self, window_size=11, mad_cutoff=5):
        if not (self.__units=='fluxes'):
            self.__convert_data_to_fluxes()

        for i, table in enumerate(self.__tables):
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
            self.__tables[i]['mask'] = ~mask

    def reset_masks(self):
        for i in range(len(self.__tables)):
            self.__tables[i]['mask'] = np.ones(len(self.__tables[i]['HJD']),
                dtype=bool)

    @property
    def light_curves(self):
        return self.__tables

    @light_curves.setter
    def light_curves(self, tables):
        for table in tables:
            if not isinstance(table, Table):
                raise ValueError("You need to provide a list of Astropy Tables.")
        self.__tables = tables

    @property
    def event_name(self):
        return self.__event_name

    @event_name.setter
    def event_name(self, value):
        if isinstance(value, str):
            self.__event_name = value
        else:
            raise ValueError("Event name has to be a string.")
    
    @property
    def event_coordinates(self):
        return self.__coordinates

    @event_coordinates.setter
    def event_coordinates(self, coordinates):
        if isinstance(coordinates, SkyCoord):
            self.__coordinates = coordinates
        else:
            raise ValueError("Event coordinates must be passed as an"
                "astropy.coordinates.SkyCoord object.")

    @property
    def units(self):
        return self.__units

    @units.setter
    def units(self, value):
        if value=='magnitudes':
            self.__convert_data_to_magnitudes()
        elif value=='fluxes':
            self.__convert_data_to_fluxes()
        else:
            raise ValueError("The only to options for units are 'magnitudes'"
                "or 'fluxes'.")

    
class OGLEData(Data):
    """
    Subclass of data class for dealing with OGLE data.
    """
    def __init__(
        self, 
        event_dir=None
    ):
        super(OGLEData, self).__init__(event_dir)
        with open(event_dir + '/params.dat') as f:
            lines = f.readlines() 
            self._Data__event_name = lines[0][:-1]
            RA = lines[4][15:-1]
            Dec = lines[5][15:-1]
            self._Data__coordinates = SkyCoord(RA, Dec, 
                unit=(u.hourangle, u.deg, u.arcminute))
        self.__load_data(event_dir)

    def __load_data(self, event_dir):
        """Returns a table with raw data."""
        t = Table.read(event_dir + '/phot.dat', format='ascii')

        # Remove additional columns
        t.columns[0].name = 'HJD'
        t.columns[1].name = 'mag'
        t.columns[2].name = 'mag_err'
        t.keep_columns(('HJD', 'mag', 'mag_err'))

        # Add mask column
        mask = Table.Column(np.ones(len(t['HJD']), dtype=bool), name='mask', 
            dtype=bool)
        t.add_column(mask)  # Insert before the first table column

        # Add 2450000 if necessary
        if(t['HJD'][0] < 2450000):
            t['HJD'] += 2450000

        t.meta = {'filter':'I', 'observatory':'OGLE'}
        self._Data__tables.append(t)


class MOAData(Data):
    """Subclass of data class for handling with MOA datasets."""
    def __init__(
        self, 
        event_path=None, 
        index_path=None
    ):
        super(MOAData, self).__init__(event_path)
        self._Data__units = 'fluxes'

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
        self._Data__event_name = 'MOA-' + key_value.iloc[0]
        self._Data__load_data(event_path)

    def __load_data(self, event_path):
        """Returns dataframe with raw data."""
        # I'm not sure that time for MOA data is in HJD
        with open(event_path) as f:
            contents = f.readlines()
            processed = ''
            for i in range(len(contents)):
                processed += re.sub("\s+", ",", contents[i].strip()) + '\n' 
            t = Table.read(processed, format='ascii')
            t.keep_columns(['col1', 'col2', 'col3'])
            t.rename_column('col1', 'HJD')
            t.rename_column('col2', 'flux')
            t.rename_column('col3', 'flux_err')
            t.meta = {'filter':'I', 'observatory':'MOA'}

            # Remove the random rows with zero time and negative time
            t = t[t['HJD'] > 0]

            # Add mask column
            mask = Table.Column(np.ones(len(t['HJD']), dtype=bool), name='mask', 
                dtype=bool)
            t.add_column(mask)  # Insert before the first table column

        self._Data__tables.append(t)


class KMTData(Data):
    """Subclass of data class for dealing with OGLE data."""
    def __init__(
        self, 
        event_dir=None
    ):
        super(KMTData, self).__init__(event_dir)
        self.__load_data(event_dir)
        self._Data__units = 'fluxes'

    def __load_data(self, event_dir):
        """Returns a table with raw data."""
        t1 = Table.read(event_dir + '/KMTA01_I.diapl', format='ascii')
        t1['col1'] += 2450000
        t1.keep_columns(('col1', 'col2', 'col3'))
        t1.rename_column('col1', 'HJD')
        t1.rename_column('col2', 'flux')
        t1.rename_column('col3', 'flux_err')
        t1.meta = {'filter':'I', 'observatory':'KMTA'}

        t2 = Table.read(event_dir + '/KMTC01_I.diapl', format='ascii')
        t2['col1'] += 2450000
        t2.keep_columns(('col1', 'col2', 'col3'))
        t2.rename_column('col1', 'HJD')
        t2.rename_column('col2', 'flux')
        t2.rename_column('col3', 'flux_err')
        t2.meta = {'filter':'I', 'observatory':'KMTC'}

        t3 = Table.read(event_dir + '/KMTS01_I.diapl', format='ascii')
        t3['col1'] += 2450000
        t3.keep_columns(('col1', 'col2', 'col3'))
        t3.rename_column('col1', 'HJD')
        t3.rename_column('col2', 'flux')
        t3.rename_column('col3', 'flux_err')
        t3.meta = {'filter':'I', 'observatory':'KMTS'}

        self._Data__tables = [t1, t2, t3]

        for t in  self._Data__tables:
            # Add mask column
            mask = Table.Column(np.ones(len(t['HJD']), dtype=bool), name='mask', 
                dtype=bool)
            t.add_column(mask)  # Insert before the first table column


class NASAExoArchiveData(Data):
    """Subclass of data class for dealing with data from NASA Exo Archive."""
    def __init__(
        self, 
        event_dir=None
    ):
        super(NASAExoArchiveData, self).__init__(event_dir)
        self.__load_data(event_dir)
        self._Data__units = 'magnitudes'

    def __load_data(self, event_dir):
        """Returns tables with raw data."""
        count = 0
        for file in os.listdir(event_dir):
            if file.endswith(".tbl"):
                t = Table.read(os.path.join(event_dir, file), format='ascii')

                if (t.colnames[0]=='JD'):                
                    t.rename_column('JD', 'HJD')
                elif (t.colnames[0]=='HJD'):
                    pass
                else:
                    raise ValueError("No column named HJD or JD.")

                if (t.colnames[1]=='Relative_Flux'):
                    m, m_err = self.__fluxes_to_magnitudes(t['Relative_Flux'],
                        t['Relative_Flux_Uncertainty'])
                    t['Relative_Flux'] = m
                    t['Relative_Flux_Uncertainty'] = m_err
                    t.rename_column('Relative_Flux', 'mag')
                    t.rename_column('Relative_Flux_Uncertainty', 'mag_err')
                    t.keep_columns(['HJD', 'mag', 'mag_err'])
                elif (t.colnames[1]=='RELATIVE_MAGNITUDE'):
                    t.rename_column('RELATIVE_MAGNITUDE', 'mag')
                    t.rename_column('MAGNITUDE_UNCERTAINTY', 'mag_err')
                    t.keep_columns(['HJD', 'mag', 'mag_err'])
                else:
                    raise ValueError("No columns specifying flux or magnitude.")
                
                info = t.meta['keywords']
                
                # Save coordinates of event, check they're consistent between datasets
                if (count==0):
                    ra = info['RA']['value']
                    dec = info['DEC']['value']
                    self.__coordinates = SkyCoord(ra, dec)        
                elif(ra!=info['RA']['value'] or\
                         dec!=info['DEC']['value']):
                    raise ValueError("Event coordinates don't match between"
                         "different datasets. ")

                # Save event name
                if (count==0):
                    self.__event_name = info['STAR_ID']['value']
                elif (self.__event_name!=info['STAR_ID']['value']):
                    self.__event_name += info['keywords']['STAR_ID']['value']
                    
                # Check that all times are HJD in epoch J2000.0    
                if (info['EQUINOX']['value']!="J2000.0"):
                    raise ValueError("Equinox for the dataset ", 
                    info['OBSERVATORY_SITE']['value'], "is not J2000.")
                if (info['TIME_REFERENCE_FRAME']['value']!="Heliocentric JD"):
                    raise ValueError("Time reference frame for ", 
                    info['OBSERVATORY_SITE']['value'],
                     "is not HJD.")
                
                # Save information about observatory name and filter used
                t.meta = {'observatory':info['OBSERVATORY_SITE']['value'],
                    'filter':info['TIME_SERIES_DATA_FILTER']['value']}

                t = Table(t, masked=False)

                # Add mask column
                mask = Table.Column(np.ones(len(t['HJD']), dtype=bool),
                    name='mask', dtype=bool)
                t.add_column(mask)  # Insert before the first table column

                self._Data__tables.append(t)

                count = count + 1