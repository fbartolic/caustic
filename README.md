`caustic` is a code designed for Bayesian modeling of single-lens gravitational microlensing events using [Hamiltonian Monte Carlo](http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html). It is built on top of 
[PyMC3](https://docs.pymc.io/), [exoplanet](https://exoplanet.dfm.io/en/latest/), and [Astropy](http://www.astropy.org/).

Although there are now several open-source codes for modeling microlensing events, none 
of those use gradient-based samplers such as Hamiltonian Monte Carlo. `caustic` is still very much in early development and lacking documentation, I hope to release a packaged version before mid 2019. 
