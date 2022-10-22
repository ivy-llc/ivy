# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy_tests.test_array_api.array_api_tests.hypothesis_helpers as hypothesis_helpers


# Helpers #
# ------- #


def _st_tuples_or_int(n_pairs):
    return st.one_of(
        hypothesis_helpers.tuples(
            st.tuples(
                st.integers(min_value=1, max_value=4),
                st.integers(min_value=1, max_value=4),
            ),
            min_size=n_pairs,
            max_size=n_pairs,
        ),
        helpers.ints(min_value=1, max_value=4),
    )


@st.composite
def _pad_helper(draw):
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            ret_shape=True,
            min_num_dims=1,
        )
    )
    ndim = len(shape)
    pad_width = draw(_st_tuples_or_int(ndim))
    stat_length = draw(_st_tuples_or_int(ndim))
    constant_values = draw(_st_tuples_or_int(ndim))
    end_values = draw(_st_tuples_or_int(ndim))
    return dtype, value, pad_width, stat_length, constant_values, end_values


@handle_cmd_line_args
@given(
    dtype_and_input_and_other=_pad_helper(),
    mode=st.sampled_from(
        [
            "constant",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
        ]
    ),
    reflect_type=st.sampled_from(["even", "odd"]),
    num_positional_args=helpers.num_positional_args(fn_name="pad"),
)
def test_pad(
    *,
    dtype_and_input_and_other,
    mode,
    reflect_type,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    (
        dtype,
        value,
        pad_width,
        stat_length,
        constant_values,
        end_values,
    ) = dtype_and_input_and_other
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="pad",
        ground_truth_backend="numpy",
        input=value[0],
        pad_width=pad_width,
        mode=mode,
        stat_length=stat_length,
        constant_values=constant_values,
        end_values=end_values,
        reflect_type=reflect_type,
        out=None,
    )
