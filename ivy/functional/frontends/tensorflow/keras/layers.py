import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def flatten(x,/,*,copy=None,start_dim=0, end_dim=-1, order='C', out=None):
    ivy.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
    if (order in ["K","A","F"]):
        return ivy.flatten(x,copy=copy,start_dims=start_dim,end_dims=end_dim,order="F",out=out)
    else:
         return ivy.flatten(x,copy=copy,start_dims=start_dim,end_dims=end_dim,order="C",out=out)