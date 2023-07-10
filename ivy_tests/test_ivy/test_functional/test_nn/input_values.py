import numpy as np
from ivy import np as ivy_np
from ivy.core.container import Container

def numpy_layer_norm(x, normalized_idxs, scale, offset, eps):
    # Convert inputs to Ivy containers
    x_ivy = Container(x)
    normalized_idxs_ivy = Container(normalized_idxs)
    scale_ivy = Container(scale)
    offset_ivy = Container(offset)

    # Call the Ivy function
    y_ivy = ivy_np.layer_norm(x_ivy, normalized_idxs_ivy, scale_ivy, offset_ivy, eps)

    # Convert the result back to NumPy array
    y = ivy_np.to_numpy(y_ivy)

    return y

@given(input_strategy())
def test_code(input_data):
    # Call the NumPy frontend function
    numpy_result = numpy_layer_norm(**input_data)

    # Call the Ivy test function
    ivy_result = test_layer_norm(**input_data)

    # Compare the results
    assert np.allclose(numpy_result, ivy_result)

