import mxnet as mx
from typing import Union, Tuple, Optional, List

def prod(x: mx.nd.NDArray,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[mx.np.dtype] = None,
         keepdims: bool = False)\
        -> mx.nd.NDArray:

    if dtype == None and mx.np.issubdtype(x.dtype,mx.np.integer):
        if mx.np.issubdtype(x.dtype,mx.np.signedinteger) and x.dtype in [mx.np.int8,mx.np.int16,mx.np.int32]:
            dtype = mx.np.int32
        elif mx.np.issubdtype(x.dtype,mx.np.unsignedinteger) and x.dtype in [mx.np.uint8,mx.np.uint16,mx.np.uint32]:
            dtype = mx.np.uint32
        elif x.dtype == mx.np.int64: 
            dtype = mx.np.int64
        else:
            dtype = mx.np.uint64
    
    return mx.np.prod(x,axis,dtype,keepdims)