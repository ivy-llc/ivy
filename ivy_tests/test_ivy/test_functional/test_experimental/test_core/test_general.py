# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@st.composite
def _reduce_helper(draw):
    dtype = draw(helpers.get_dtypes("valid", full=False))

    if dtype[0] == "bool":
        func = draw(st.sampled_from([ivy.logical_and, ivy.logical_or]))
    else:
        func = draw(
            st.sampled_from([ivy.add, ivy.max, ivy.min, ivy.multiply])
        )
    init_value = draw(
        st.sampled_from(
            [
                [-float("inf")],
                [float("inf")],
                draw(
                    helpers.dtype_and_values(
                        dtype=dtype,
                        shape=(),
                    )
                )[1],
            ]
        )
    )
    dtype, operand, axis = draw(
        helpers.dtype_values_axis(
            min_num_dims=1,
            dtype=dtype,
        )
    )
    return dtype, operand, init_value[0], func, axis


# reduce
@handle_test(
    fn_tree="functional.ivy.experimental.reduce",
    all_args=_reduce_helper(),
)
def test_reduce(
    *,
    all_args,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, operand, init_value, func, axis = all_args
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        operand=operand,
        init_value=init_value,
        func=func,
        axis=axis,
    )
