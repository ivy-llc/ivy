@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_exponential(size = None):
    return ivy.standard_exponential(scale = 1.0,shape = size,dtype = 'float64',method = 'zig')
