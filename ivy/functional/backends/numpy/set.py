import numpy as np
from packaging import version


def unique_values(x: np.ndarray)\
        -> np.ndarray:
    nan_count = np.count_nonzero(np.isnan(x))
    if (version.parse(np.__version__) >= version.parse('1.21.0') and nan_count > 1):
        unique = np.append(np.unique(x.flatten()), np.full(nan_count - 1, np.nan)).astype(x.dtype)
    else:
        unique = np.unique(x.flatten()).astype(x.dtype)
    return unique
