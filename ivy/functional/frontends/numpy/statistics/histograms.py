import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"1.25.2 and below": ("int64",)}, "numpy")
@to_ivy_arrays_and_back
def bincount(x, /, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)


@with_supported_dtypes(
    {
        "1.25.2 and below": (
            "int64",
            "float64",
        )
    },
    "numpy",
)
@to_ivy_arrays_and_back
def histogram(a, bins=10, range=None, density=None, weights=None):
    if range:
        data_min, data_max = range
    else:
        data_min = min(a)
        data_max = max(a)

    bin_edges = ivy.linspace(data_min, data_max, bins + 1)
    bin_counts = ivy.zeros(bins, dtype="float64")

    if range:
        valid_indices = (a >= data_min) & (a <= data_max)
        a = a[valid_indices]
        if weights is not None:
            weights = weights[valid_indices]

    bin_indices = ((ivy.array(a) - data_min) / (data_max - data_min) * bins).astype(int)
    bin_indices = ivy.clip(bin_indices, 0, bins - 1)

    if weights is None:
        bin_counts += ivy.bincount(bin_indices, minlength=bins)
    else:
        bin_counts += ivy.bincount(bin_indices, weights=weights, minlength=bins)

    if density:
        total_weight = ivy.sum(bin_counts)
        bin_width_normalized = (
            (data_max - data_min) / bins
            if density is None
            else (data_max - data_min) / (total_weight * bins)
        )
        bin_counts /= total_weight * bin_width_normalized

    return bin_counts.astype("int64"), bin_edges
