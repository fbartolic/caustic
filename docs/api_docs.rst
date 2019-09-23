.. module:: caustic 
.. _api:

API Documentation
=================

Data 
----
    
.. autoclass:: caustic.data.Data
    :members:
    :inherited-members:

.. autoclass caustic.data.OGLEData
   :members:
   :inherited-members:
.. autoclass caustic.data.MOAData
   :members:
   :inherited-members:
.. autoclass caustic.data.KMTData
   :members:
   :inherited-members:
.. autoclass caustic.data.NASAExoArchiveDatak
   :members:
   :inherited-members:

Models 
------

.. autoclass:: caustic.models.SingleLensModel
   :members:

Trajectory
----------

.. autoclass:: caustic.trajectory.Trajectory
   :members:

Utils
-----

.. autofunction:: caustic.estimate_t0
.. autofunction:: caustic.compute_source_mag_and_blend_fraction
.. autofunction:: caustic.plot_model_and_residuals
.. autofunction:: caustic.plot_map_model_and_residuals
.. autofunction:: caustic.plot_trajectory_from_samples
.. autofunction:: caustic.sample_with_dynesty
.. autofunction:: caustic.compute_invgamma_params 