import numpy as np
import copy

from caustic.data import OGLEData

np.random.seed(42)

event = OGLEData("../../data/OGLE-2017-BLG-0324")

def test_convert_data_to_fluxes():
    """Tests the consistency between conversion to fluxes and magnitudes."""
    event_initial = copy.deepcopy(event)
    event.units = 'fluxes'
    event.units = 'magnitudes'

    # Iterate over bands
    for i in range(len(event_initial.light_curves)):
        assert  np.allclose(
                event_initial.light_curves[i]['mag'], 
                event.light_curves[i]['mag']
        )
        assert  np.allclose(
                event_initial.light_curves[i]['mag_err'], 
                event.light_curves[i]['mag_err']
        )

def test_magnitudes_to_fluxes():
    """
    Tests weather the analytic expressions for the mean and root-variance 
    of the flux random variables are correct by simulating samples from a 
    normal distribution.
    """
    # Iterate over bands
    for i, table in enumerate(event.light_curves):
        m = np.array(table['mag'])
        sig_m = np.array(table['mag_err'])

        # Sample multivariate normal distribution with those parameters
        m_samples = np.random.multivariate_normal(m, np.diag(sig_m**2),
                size=10000)

        F_samples = 10**(-(m_samples - 22)/2.5)

        mu_F = np.mean(F_samples, axis=0)
        std_F = np.std(F_samples, axis=0)

        event_copy = copy.deepcopy(event)
        event_copy.units = 'fluxes'

        assert np.allclose(
                mu_F, np.array(event_copy.light_curves[i]['flux']), rtol=1.e-02
        )
        assert np.allclose(
                std_F, np.array(event_copy.light_curves[i]['flux_err']), 
                rtol=1.e-01
        )

def test_get_standardized_data():
    """Standardized data should have zero median and unit std dev."""
    std_tables = event.get_standardized_data()

    # Iterate over bands
    for i, table in enumerate(std_tables):
        assert np.allclose(
           np.median(table['flux']), 0.
        )
        assert np.allclose(
           np.std(table['flux']), 1.
        )