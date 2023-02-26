# local
from collections import namedtuple
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def unique(
    array, /, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    results = ivy.unique_all(array)

    fields = ["values"]
    if return_index:
        fields.append("indices")
    if return_inverse:
        fields.append("inverse_indices")
    if return_counts:
        fields.append("counts")

    Results = namedtuple("Results", fields)

    values = [results.values]
    if return_index:
        values.append(results.indices)
    if return_inverse:
        # numpy flattens inverse indices like unique values
        # if axis is none, so we have to do it here for consistency
        values.append(
            results.inverse_indices
            if axis is not None
            else ivy.flatten(results.inverse_indices)
        )
    if return_counts:
        values.append(results.counts)

    return Results(*values)
