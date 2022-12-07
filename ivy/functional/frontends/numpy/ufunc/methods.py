#local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays

@inputs_to_ivy_arrays
def at(
    ufunc,
    array,
    indices,
    b=None,
    /):
  if b:
    for index in indices:
      array[index] = ufunc(array[index], b)
  else:
    for index in indices:
      array[index] = ufunc(array[index])
