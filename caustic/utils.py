import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import exoplanet as xo
import pymc3 as pm

from data import OGLEData

mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=False)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16

def qq_plot(ax, residuals):
    """Plots quantile-quantile plot of residuals vs draws from a Gaussian."""
    percentiles = np.linspace(0, 100, 500)
    quantile_residuals = np.percentile(residuals, percentiles)
    quantile_model = np.percentile(np.random.normal(size=10000), percentiles)

    ax.scatter(quantile_model, quantile_residuals, marker='o', color='black',
        alpha=0.3)
    ax.grid()
    x = np.linspace(quantile_model[0], quantile_model[-1])
    # ax.plot(x, x, color='black', linestyle='dashed', alpha=0.8)
    ax.set_aspect(1)
    ax.set_xlabel("Modeled residuals")
    ax.set_ylabel("Measured residuals")


#def plot_mock_data(ax, data, pm_model, trace):
#    """Plots mock data evaluated at original observation times."""
#    median_params_dict = {}
#    for i in range(len(pm_model.vars)):
#        median_params_dict[str(pm_model.vars[i])] = \
#            np.percentile(trace[pm_model.vars[i]], [16, 50, 84], 
#            axis=0)[1]
#    
#    df = data.get_standardized_data()
#    t = df['HJD - 2450000'].values
#
#    with pm_model:
#        # Evaluate residuals
#        F_base = pm_model.F_base
#        u0 = pm_model.u0
#        t0 = pm_model.t0
#        Delta_F = pm_model.Delta_F
#        tE = pm_model.tE
#        u = T.sqrt(u0**2 + ((t - t0)/tE)**2)
#        A = lambda u: (u**2 + 2)/(u*T.sqrt(u**2 + 4))
#        mean_func = Delta_F*(A(u) - 1)/(A(u0) - 1) + F_base
#
#        mean_function = xo.eval_in_model(mean_func, 
#            median_params_dict)
#        
#    F_err = df['I_flux_err'].values*np.median(trace['K'])
#
#    mock_data = np.random.multivariate_normal(mean=mean_function,
#        cov=np.diag(F_err))
#
#    ax[0].errorbar(t, mock_data, F_err, fmt='.', color='black', label='Data', 
#        ecolor='#686868')
#    ax[0].grid(True)
#    ax[0].set_title(data.event_name + 'mock')
#    ax[0].set_xlabel(df.columns[0])
#    ax[0].set_ylabel(df.columns[1])
#
#    # Plot residuals
#    df = event.get_standardized_data()
#    residuals = mock_data - mean_function
#    ax[1].errorbar(t, residuals, F_err, 
#        fmt='.', color='black', ecolor='#686868')
#    ax[1].grid(True)

def plot_model_and_residuals(ax, event, pm_model, trace_path):
    # Load standardized data
    tables = event.get_standardized_data()
#    t_grid = np.linspace(tables[0]['HJD'][0], 
#            tables[0]['HJD'][-1], 2000)
    t_grid = np.linspace(7950,
            8050, 500)


    # Compute model predictions on a fine grid and at observed times
    with pm_model as model_instance:
        trace = pm.load_trace(trace_path) 
        pred = model_instance.evaluate_model_on_grid(trace, t_grid, 
            n_samples=50)
        pred_at_observed_times = [model_instance.evaluate_model_on_grid(trace,
            table['HJD'], n_samples=100) for table in tables]

    # Plot data
    event.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')

    n_bands = len(tables)

    # Plot predictions for various samples
    for n in range(n_bands): # iterate over bands
        for i in range(len(pred[n, :, 0])):
            ax[0].plot(t_grid, pred[n, i, :], color='C' + str(n), 
                alpha=0.2)

    # Calculate and plot residuals
    for n in range(n_bands): # iterate over bands
        quantile_predictions = np.percentile(pred_at_observed_times[n],
            [16, 50, 84], axis=1)

        residuals =  tables[n]['flux'] - quantile_predictions[1, n, :]
        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='.', color='C' + str(n))
        ax[1].grid(True)

def plot_map_model_and_residuals(ax, event, pm_model, map_point):
    # Load standardized data
    tables = event.get_standardized_data()
    t_grid = np.linspace(tables[0]['HJD'][0], 
            tables[0]['HJD'][-1], 2000)

    # Compute model predictions on a fine grid and at observed times
    with pm_model as model_instance:
        pred = model_instance.evaluate_map_model_on_grid(t_grid, map_point)
        pred_at_observed_times = [model_instance.evaluate_map_model_on_grid(
            table['HJD'], map_point) for table in tables]

    # Plot data
    event.plot_standardized_data(ax[0])
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('HJD - 2450000')
    ax[1].set_ylabel('Residuals')

    n_bands = len(tables)

    # Plot predictions for various samples
    for n in range(n_bands): # iterate over bands
        ax[0].plot(t_grid, pred[n, :], color='C' + str(n), alpha=0.5)
        print(pred[n, :])

    # Calculate and plot residuals
    for n in range(n_bands): # iterate over bands
        residuals =  tables[n]['flux'] - pred_at_observed_times[n][n]
        ax[1].errorbar(tables[n]['HJD'], residuals, tables[n]['flux_err'],
            fmt='.', color='C' + str(n))
        ax[1].grid(True)

def plot_violin_plots(ax, pm_models, trace_paths, parameter_names):
    for idx, model in enumerate(pm_models):
        with pm_model as model_instance:
            trace = pm.load_trace(trace_path) 

    return 0