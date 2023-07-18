# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# TODO fix instance method and report using force_int_axis returns a None axis sometims
@handle_test(
    fn_tree="functional.ivy.experimental.tensors.unfold",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        allow_neg_axes=False,
        force_int_axis=True,
    ),
)
def test_unfold(*, dtype_values_axis, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, axis = dtype_values_axis
    if axis is None:
        axis = 0
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        mode=axis,
    )


@st.composite
def _fold_data(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=3
        )
    )
    mode = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    reduced_dims = int(ivy.prod(shape[0:mode]) * ivy.prod(shape[mode + 1 :]))
    unfolded_shape = (shape[mode], reduced_dims)
    dtype, input = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"), shape=unfolded_shape
        )
    )
    return dtype, input, shape, mode


# TODO fix instance method
@handle_test(
    fn_tree="functional.ivy.experimental.tensors.fold",
    data=_fold_data(),
)
def test_fold(*, data, test_flags, backend_fw, fn_name, on_device):
    input_dtype, input, shape, mode = data
    test_flags.instance_method = False
    helpers.test_function(
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        input_dtypes=input_dtype,
        input=input,
        mode=mode,
        shape=shape,
    )
