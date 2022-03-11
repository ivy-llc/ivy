
import mxnet as mx

def prod(x: mx.nd.NDArray,
         axis: Union[int, Tuple[int]] = None,
         dtype: mx.dtype = None,
         keepdims: bool = False)\
        -> mx.nd.NDArray:
    return mx.np.prod(x,axis,dtype,keepdims)