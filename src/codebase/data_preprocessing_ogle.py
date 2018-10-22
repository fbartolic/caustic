import numpy as np

def magnitudes_to_fluxes(m, sig_m, m0):
    """Given the mean and the standard deviation of a magnitude, assumed to be normally
    distributed, and a reference magnitude m0, this function returns the mean and the 
    standard deviation of a Flux, which is log-normally distributed."""

    # Calculate the mean and std. deviation for lnF
    e = np.exp(1)
    mu_lnF = m0/(2.5*np.log10(e)) - m
    sig_lnF = sig_m/(2.5*np.log10(e))

    # If lnF is gaussian distributed F is log-normal distributed
    # approximate the log-normal distribution with a Gaussian 
    mu_F = np.exp(mu_lnF + 0.5*sig_lnF**2)
    sig_F = np.sqrt((np.exp(sig_lnF**2) - 1)*np.exp(2*mu_lnF + sig_lnF**2))

    return mu_F, sig_F    

def process_data(time, mag, mag_std, standardize=True, mag_zeropoint=22):  
    """This function takes the microlensing event data from OGLE and converts
    it to a format suitable for modelling.  
    
    Parameters
    ----------
    time :  ndarray
        Timestamp in HJD.
    mag : ndarray
        Flux measurements in magnitudes.
    mag_std : ndarray
        The standard error of corresponding magnitudes.
    standardize : bool
        If true, the data is rescaled to unit variance and zero median 
        according to Z = (F - med(F)/std(F))
    mag_zeropoint: float
        Sets the magnitude m0 at which the Flux is equal to zero.
    
    Returns
    -------
    tuple 
        Tuple of numpy arrays (time, F, sig_F) where time is now expressed in 
        days since HJD2450000, F is the flux, and sig_F the corresponding flux
        standard error.
    """

    time = time - 2450000
    
    F, sig_F = magnitudes_to_fluxes(mag, mag_std, 22.)

    if standardize==True:
        # Subtract the median from the data such that baseline is at approx 
        # zero, rescale the data such that it has unit variance
        F_r = (F - np.median(F))/np.std(F)
        sig_F_r = sig_F/np.std(F)
        return time, F_r, sig_F_r
    else:
        return time, F, sig_F 
