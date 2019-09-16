.. module:: caustic

caustic
=======

caustic is a code designed for Bayesian modeling of single-lens gravitational microlensing 
events using the `PyMC3 <https://docs.pymc.io/>`_ probabilistic programming language which enables
the use of gradient based inference algorithms such as 
`Hamiltonian Monte Carlo <http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html>`_. 
It is built on top of `theano <http://deeplearning.net/software/theano/>`_, 
`exoplanet <https://exoplanet.dfm.io/en/latest/>`_ and `Astropy <http://www.astropy.org/>`_.

The code is being publicly developed in a `repository on GitHub 
<https://github.com/fbartolic/caustic>`_.

.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/fbartolic/caustic/blob/master/LICENSE

.. raw:: html

    <br>

.. toctree::
   :maxdepth: 2
   :caption: Installation 

   install

.. toctree::
   :maxdepth: 2
   :caption: Tutorials 

   examples/getting_started 
   examples/gaussian_processes
   examples/annual_parallax
   examples/annual_parallax_dynesty

.. toctree::
   :maxdepth: 2
   :caption: Usage 

   api_docs