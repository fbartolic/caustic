[![tests](https://github.com/fbartolic/caustic/actions/workflows/tests.yml/badge.svg)](https://github.com/fbartolic/caustic/actions/workflows/tests.yml)


# Overview 
`caustic` is a [JAX](https://github.com/google/jax)-based library for modeling photometric and 
astrometric **single-lens microlensing events**. Thanks to JAX, `caustic`  enables enables the computation of *exact* gradients of all quantities of interest with respect to the input parameters through the use of [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html). `caustic` runs on both CPUs and GPUs and it is extremely fast.

## Installation
`caustic` is still being actively developed and is not yet released on PyPI. To install the development version, clone this repository, 
create a new `conda` environment, `cd` into the repository and run 
```python
conda env update --file environment.yml && pip install .
```

## References
- paper coming soon!
