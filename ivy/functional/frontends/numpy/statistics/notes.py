# global
import numpy as np
import ivy.functional.frontends.numpy
from ivy.functional.frontends.numpy import from_zero_dim_arrays_to_scalar
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def quantile(a,
             q,
             axis=None,
             out=None,
             overwrite_input=False,
             method="linear",
             keepdims=False,
             *,
             interpolation=None):

    axis = tuple(axis) if isinstance(axis, list) else axis



    index = (a.size * (q / 100)) - 1
    previous = np.floor(index)  # <---------------------------------------------- Numpy*
    if method == "inverted_cdf":
        if index < 1:
            inverted_cdf = 1
        elif index % 1 == 0:  # if g = 0 ; then take i
            inverted_cdf = previous
        else:  # if g > 0 ; then take j
            inverted_cdf = (previous + 1)
        return int(inverted_cdf)
    elif method == "averaged_inverted_cdf":
        if index < 1:
            averaged_inverted_cdf = 1
        elif index % 1 == 0:  # if g = 0 ; then average between bounds
            averaged_inverted_cdf = (previous + (previous + 1)) / 2
        else:  # if g > 0 ; then take j
            averaged_inverted_cdf = (previous + 1)
        return averaged_inverted_cdf
    elif method == "closest_observation":
        index_co = (a.size * (q / 100)) - 1 - 0.5
        previous_co = np.floor(index_co)  # <---------------------------------------------- Numpy*
        if index_co < 1:
            closest_observation = 1
        elif index_co % 1 == 0 and (index_co % 2) != 0:  # if g = 0 and index is even ; then take i
            closest_observation = previous_co
        elif index_co % 1 == 0 and (index_co % 2) == 0:  # if g = 0 and index is odd ; then take j
            closest_observation = (previous_co + 1)
        else:  # if g > 0 ; then take j
            closest_observation = (previous_co + 1)
        print("closest_observation = ", int(closest_observation))
    elif method == "interpolated_inverted_cdf":
        interpolated_inverted_cdf = a.size * (q / 100) + (0 + (q / 100) * (1 - 0 - 1)) - 1
        return interpolated_inverted_cdf
    elif method == "hazen":
        hazen = a.size * (q / 100) + ((1 / 2) + (q / 100) * (1 - (1 / 2) - (1 / 2))) - 1
        return hazen
    elif method == "weibull":
        weibull = a.size * (q / 100) + (0 + (q / 100) * (1 - 0 - 0)) - 1
        return weibull
    elif method == "linear":
        keepdims = "linear"
        return keepdims
    elif method == "median_unbiased":
        median_unbiased = a.size * (q / 100) + ((1 / 3) + (q / 100) * (1 - (1 / 3) - (1 / 3))) - 1
        return median_unbiased
    elif method == "normal_unbiased":
        normal_unbiased = a.size * (q / 100) + ((3 / 8) + (q / 100) * (1 - (3 / 8) - (3 / 8))) - 1
        return normal_unbiased
    elif method == "lower":
        keepdims = "lower"
        return keepdims
    elif method == "higher":
        keepdims = "higher"
        return keepdims
    elif method == "midpoint":
        keepdims = "midpoint"
        return keepdims
    elif method == "nearest":
        keepdims = "nearest"
        return keepdims

    ret = ivy.quantile(a, q, axis=axis, keepdims=keepdims, out=out,
                       interpolation=interpolation)

    return ret


a = np.array([[1, 2, 5, 6], [4, 7, 3, 1]])
q = 60
b = np.percentile(a, q, method="weibull")
