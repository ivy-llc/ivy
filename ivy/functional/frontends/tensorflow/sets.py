import ivy
from ivy.functional.frontends.tensorflow import check_tensorflow_casting
from ivy.functional.frontends.tensorflow.func_wrapper import (to_ivy_arrays_and_back)


@to_ivy_arrays_and_back
def difference(a, b, aminusb=True, validate_indices=True):
    a, b = check_tensorflow_casting(a, b)
    return ivy.difference(a, b, aminusb=aminusb, validate_indices=validate_indices)


@to_ivy_arrays_and_back
def intersection(a, b, validate_indices=True):
    a, b = check_tensorflow_casting(a, b)
    return ivy.intersection(a, b, validate_indices=validate_indices)


@to_ivy_arrays_and_back
def union(a, b, validate_indices=True):
    a, b = check_tensorflow_casting(a, b)
    return ivy.union(a, b, validate_indices=validate_indices)
