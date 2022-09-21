import ivy
from jax.numpy import linalg

from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import _remove_jaxarray


def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


 #adding a max_power function

def matrix_power(m,n):
    #m is the input marix that is created by a numpy
    #n is the output exponential power
    # return will be n*m which is a ndarray or a matrix object
    m= _remove_jaxarray(m)
    return JaxArray(linalg.matrix_power(m,n))



