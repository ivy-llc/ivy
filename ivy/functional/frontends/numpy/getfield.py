import numpy as np

def getfield(arr, field_name):
    """
    Get the specified field from a structured array.

    Parameters:
        arr (numpy.ndarray): A structured array.
        field_name (str): The name of the field to retrieve.

    Returns:
        numpy.ndarray: The field array.
    """
    if not isinstance(arr, np.ndarray) or not arr.dtype.fields:
        raise ValueError("Input must be a structured array.")

    if field_name not in arr.dtype.fields:
        raise ValueError(f"Field '{field_name}' does not exist in the input array.")

    return arr[field_name]
