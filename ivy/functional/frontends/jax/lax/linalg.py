import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)


@to_ivy_arrays_and_back
def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


@to_ivy_arrays_and_back
def eigh(x, /, *, lower=True, symmetrize_input=True, sort_eigenvalues=True):
    UPLO = "L" if lower else "U"

    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.eigh(x, UPLO=UPLO)

  @to_ivy_arrays_and_back
    def all_gather(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False): 
        return ivy.all_gather(x, 'i', axis_index_groups=[[0, 2], [3, 1]])      
     
  @to_ivy_arrays_and_back
        def all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None, tiled=False): 
            return ivy.np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)
        
  @to_ivy_arrays_and_back 
        def psum(x, axis_name, *, axis_index_groups=None):  
        if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  leaves = [lax.convert_element_type(l, np.int32)
            if dtypes.dtype(l) == np.bool_ else l for l in leaves]
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  out_flat = psum_p.bind(
      *leaves, axes=tuple(axis_name), axis_index_groups=axis_index_groups)
  return ivy.tree_util.tree_unflatten(treedef, out_flat)

@to_ivy_arrays_and_back 
def pmax(x, axis_name, *, axis_index_groups=None):
    return ivy.tree_util.tree_unflatten(treedef, out_flat)   
    
    
        
        
