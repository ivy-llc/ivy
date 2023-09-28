import paddle
import hypothesis.strategies as st
import numpy as np


# Define a Hypothesis strategy to generate valid input arguments for paddle.roll
def roll_input_strategy():
    # Import the helpers module from Hypothesis
    from hypothesis import helpers

    # Generate random dtype from valid dtypes
    dtype = st.sampled_from(helpers.get_dtypes("valid"))

    # Generate random data tensor
    shape = st.tuples(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
    )
    data = st.arrays(dtype=np.float32, shape=shape)

    # Generate random shifts value
    shifts = st.integers(min_value=-10, max_value=10)

    # Generate random axis value (list or tuple of two unique integers)
    axis = st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=2,
        max_size=2,
        unique=True,
    )

    return dtype, data, shifts, axis


# Usage example to generate input and print it
if __name__ == "__main__":
    from hypothesis import example

    input_args = roll_input_strategy().example()
    dtype, data, shifts, axis = input_args

    print("dtype:", dtype)
    print("data:", data)
    print("shifts:", shifts)
    print("axis:", axis)

    # Calculate the output by calling paddle.roll
    result = paddle.roll(data, shifts, axis)
    print("result:", result)
