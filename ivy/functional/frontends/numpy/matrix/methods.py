import numpy.core.numeric as N
import numpy as np

def asmatrix(data, dtype=None):
    return _matrix(data,dtype)

class _matrix:
  def __init__(self,y,dtype):
    self.data= np.matrix(y, dtype=dtype, copy=False)
  def min(self,axis=None, out=None):
        return N.ndarray.min(self.data, axis, out, keepdims=True)

matrix=asmatrix
