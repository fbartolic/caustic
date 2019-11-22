import numpy as np
import pymc3 as pm
import theano.tensor as T
from exoplanet import eval_in_model

from caustic import (
    construct_masked_tensor,
    get_log_likelihood_function,
    get_log_probability_function,
)

np.random.seed(42)


def test_construct_masked_tensor():
    # Create list of numpy arrays of varying lengths
    a = np.random.rand(1000)
    b = np.random.rand(600)
    c = np.random.rand(800)

    list_np = [a, b, c]

    tensor, mask = construct_masked_tensor(list_np)

    # Check if shapes are correct
    assert T.shape(tensor).eval()[0] == 3
    assert T.shape(tensor).eval()[1] == 1000

    # Convert back to numpy
    for i in range(3):
        assert np.allclose(list_np[i], tensor[i][mask[i].nonzero()].eval())


def test_get_log_likelihood_function():
    # Create trivial model
    with pm.Model() as model:
        # Define priors
        x1 = pm.HalfCauchy("x1", beta=10, shape=(2,), testval=[1.46, 2.1])
        x2 = pm.Normal("x2", 0, sigma=20, testval=2.37)
        x3 = pm.Exponential("x3", 0.2, testval=3.48)

        pm.Potential("log_likelihood", T.sum(x1) * x2 + x3)

    with model:
        loglike = get_log_likelihood_function(model.log_likelihood)

    with model:
        ll1 = eval_in_model(model.log_likelihood)

    ll2 = loglike([1.46, 2.1, 2.37, 3.48])

    assert ll1 == ll2


def test_get_log_probability_function():
    # Create trivial model
    with pm.Model() as model:
        # Define priors
        x1 = pm.Normal("x1", 5.0, 6.0, shape=(2,), testval=[1.46, 2.1])
        x2 = pm.Normal("x2", 1.5, 20, testval=2.37)
        x3 = pm.Normal("x3", 0.2, 5.4, testval=3.48)

        pm.Potential("log_likelihood", T.sum(x1) * x2 + x3)

    with model:
        logp = get_log_probability_function()

    with model:
        logp1 = eval_in_model(model.logpt)

    logp2 = logp([1.46, 2.1, 2.37, 3.48])

    assert logp1 == logp2
