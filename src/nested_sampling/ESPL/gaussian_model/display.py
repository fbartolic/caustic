import numpy as np
from matplotlib import pyplot as plt
import corner

# Load the data, plot it, and overplot regression lines
data = np.loadtxt("../../../microlensing_data/OGLE/2017/blg-0830/phot.dat")
posterior_samples = np.loadtxt('posterior_sample.txt')

mag_to_flux = lambda m: 10**(-m/2.5)
magerr_to_fluxerr = lambda m, sigm: sigm*mag_to_flux(m)

#data['HJD'] -= 2450000
sigF = magerr_to_fluxerr(data[:, 1], data[:, 2])
F = mag_to_flux(data[:, 1])

# Normalize flux units to unit interval
F_max = F.max()
F_min = F.min()
F = (F - F_min)/(F_max - F_min)
sigF = sigF/(F_max - F_min)


def pspl_flux(t, pars):
        """
        Evaluate a PSPL model at the input t values.


        Parameters
        ----------
        pars : list, array
            This should be a length-5 array or list containing the 
            parameter values (DeltaF, Fb, t0, u0, tE, rho).
        t : numeric, list, array
            The time values.

        Returns
        -------
        F : array
            The computed flux values at each input t.
        """
            
        DeltaF, Fb, t0, u0, tE, rho_s = pars

        u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
        A = lambda u: (u**2 + 2)/(u*np.sqrt(u**2 + 4))

        return DeltaF*(A(u) - 1)/(A(u0) - 1) + Fb    

def plot_data(ax, t, F, sigF):
    ax.errorbar(t, F, sigF, fmt='.', color='black')
    ax.grid(True)
    ax.set_xlabel('HJD - 2450000')
    ax.set_ylabel('$\Delta F$')
    ax.set_title('OGLE-2017-BLG-0401');

def triangle_plot(posterior_samples):
    lbls = [r'$\Delta F$', r'$F_b$', r'$t_0$', r'$u_0$', 
                                     r'$t_E$']#, r'$\rho$']
    fig = corner.corner(posterior_samples[:-2], labels=lbls)

fig, ax = plt.subplots(figsize=(16,8))

x_grid = np.linspace(data[0, 0], data[-1, 0], 2000)

# Plot data
plot_data(ax, data[:, 0], data[:, 1], 
        data[:, 2])

# Plot model in data space
for pars in posterior_samples[:100]: # only plot 100 samples
    ax.plot(x_grid, pspl_flux(x_grid, pars), 
             marker='', linestyle='-', color='C0', alpha=0.1, zorder=-10)
    
# ax.set_xlim(7780,8060)
# Triangle plot
#triangle_plot(posterior_samples)

plt.show()
