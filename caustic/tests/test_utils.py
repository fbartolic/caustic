import numpy as np
import theano.tensor as T

from caustic import construct_masked_tensor

np.random.seed(42)

def test_construct_masked_tensor():
    # Create list of numpy arrays of varying lengths
    a = np.random.rand(1000)
    b = np.random.rand(600)
    c = np.random.rand(800)
    
    list_np = [a, b, c]

    tensor, mask = construct_masked_tensor(list_np)

    # Check if shapes are correct
    assert T.shape(tensor).eval()[0]==3
    assert T.shape(tensor).eval()[1]==1000

    # Convert back to numpy
    for i in range(3):
       assert np.allclose(
           list_np[i], tensor[i][mask[i].nonzero()].eval()
       )