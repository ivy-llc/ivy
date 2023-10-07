# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _elemwise_helper(draw):
    value_strategy = st.one_of(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
        ),
        st.integers(min_value=-10000, max_value=10000),
        st.floats(min_value=-10000, max_value=10000),
    )

    dtype_and_x1 = draw(value_strategy)
    if isinstance(dtype_and_x1, tuple):
        dtype1 = dtype_and_x1[0]
        x1 = dtype_and_x1[1][0]
    else:
        dtype1 = []
        x1 = dtype_and_x1

    dtype_and_x2 = draw(value_strategy)
    if isinstance(dtype_and_x2, tuple):
        dtype2 = dtype_and_x2[0]
        x2 = dtype_and_x2[1][0]
    else:
        dtype2 = []
        x2 = dtype_and_x2

    num_pos_args = None
    if not dtype1 and not dtype2:
        num_pos_args = 2
    elif not dtype1:
        x1, x2 = x2, x1
    input_dtypes = dtype1 + dtype2

    return x1, x2, input_dtypes, num_pos_args


# --- Main --- #
# ------------ #


# ToDo: Fix this test after torch override of assert is implemented
# @handle_frontend_test(
#     fn_tree="torch._assert",
#     dtype_and_x=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("valid"),
#         num_arrays=2,
#     ),
#     test_with_out=st.just(False),
# )
# def test_torch__assert(
#     dtype_and_x,
#     on_device,
#     fn_tree,
#     frontend,
#     test_flags,
# ):
#     input_dtype, x = dtype_and_x
#     helpers.test_frontend_function(
#         input_dtypes=input_dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         condition=x[0],
#         message=x[1],
#     )


# bincount
@handle_frontend_test(
    fn_tree="torch.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
    test_with_out=st.just(False),
)
def test_torch_bincount(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        weights=None,
        minlength=0,
    )


@handle_frontend_test(
    fn_tree="torch.result_type",
    dtypes_and_xs=_elemwise_helper(),
    test_with_out=st.just(False),
)
def test_torch_result_type(
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    x1, x2, input_dtypes, num_pos_args = dtypes_and_xs
    if num_pos_args is not None:
        test_flags.num_positional_args = num_pos_args
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x1,
        other=x2,
    )
