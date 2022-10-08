import ivy
from ivy.functional.frontends.jax.func_wrapper import inputs_to_ivy_arrays


@inputs_to_ivy_arrays
def stop_gradient(x):
    return ivy.stop_gradient(x)


def custom_linear_solve(matvec, b, solve, transpose_solve=None,
                        symmetric=False, has_aux=False):
    return ivy.custom_linear_solve(matvec, b, solve, transpose_solve=None,
                                   symmetric=False, has_aux=False)
