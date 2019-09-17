import numpy as np

import caustic as ca

np.random.seed(42)

event_ogle = ca.data.OGLEData("../../data/OGLE-2017-BLG-0324")

def test_convert_data_to_fluxes():
    """Tests the consistency between conversion to fluxes and magnitudes."""
    df_initial = event.df.copy()
    event.convert_data_to_fluxes()
    event.convert_data_to_magnitudes()

    assert np.allclose(df_initial['I_mag'].values, event.df['I_mag'].values)\
         and np.allclose(df_initial['I_mag_err'].values, 
            event.df['I_mag_err'].values) 

def test_magnitudes_to_fluxes():
    """
    Tests weather the analytic expressions for the mean and root-variance 
    of the flux random variables are correct by simulating samples from a 
    normal distribution.
    """
    m = event.df['I_mag'].values
    sig_m = event.df['I_mag_err'].values

    m_samples = np.random.multivariate_normal(m, np.diag(sig_m**2), size=10000)

    F_samples = 10**(-(m_samples - 22)/2.5)

    mu_F = np.mean(F_samples, axis=0)
    std_F = np.std(F_samples, axis=0)

    event.convert_data_to_fluxes()
    print(std_F[:5])
    print(event.df['I_flux_err'].values[:5])
    assert np.allclose(mu_F, event.df['I_flux'].values, rtol=1.e-02) and\
        np.allclose(std_F, event.df['I_flux_err'].values, rtol=1.e-01)
