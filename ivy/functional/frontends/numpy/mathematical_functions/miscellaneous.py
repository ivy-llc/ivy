from math import inf
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float

@from_zero_dim_arrays_to_float
def clip(a, 
         a_min, 
         a_max, 
         /,
         out=None,
         *,
         where=True,
         casting="same_kind",
         order="k",
         dtype=None,
         subok=True,):
    
    if not dtype:
        dtype = a.dtype

    ret = ivy.minimum(a_max, ivy.maximum(a, a_min), out=out)
        
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    ret = ivy.astype(ivy.array(ret), ivy.as_ivy_dtype(dtype), out=out)

    return ret
