# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# helpers
@st.composite
def _get_dtype_input_and_vectors(draw, with_input=False, same_size=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = dim_size1 if same_size else draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    vec1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size1,), min_value=2, max_value=5
        )
    )
    vec2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size2,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, vec1, vec2
    return dtype, vec1, vec2


@st.composite
def _get_dtype_input_and_matrices(draw, with_input=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    mat1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size1, shared_size), min_value=2, max_value=5
        )
    )
    mat2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(shared_size, dim_size2), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, mat1, mat2
    return dtype, mat1, mat2


@st.composite
def _get_dtype_and_3dbatch_matrices(draw, with_input=False, input_3d=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    batch_size = draw(helpers.ints(min_value=2, max_value=4))
    mat1 = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(batch_size, dim_size1, shared_size),
            min_value=2,
            max_value=5,
        )
    )
    mat2 = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(batch_size, shared_size, dim_size2),
            min_value=2,
            max_value=5,
        )
    )
    if with_input:
        if input_3d:
            input = draw(
                helpers.array_values(
                    dtype=dtype[0],
                    shape=(batch_size, dim_size1, dim_size2),
                    min_value=2,
                    max_value=5,
                )
            )
            return dtype, input, mat1, mat2
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, mat1, mat2
    return dtype, mat1, mat2


@st.composite
def _get_dtype_input_and_mat_vec(draw, *, with_input=False, skip_float16=False):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=True))

    if skip_float16:
        dtype = tuple(set(dtype).difference({"float16"}))

    dtype = draw(st.sampled_from(dtype))

    mat = draw(
        helpers.array_values(
            dtype=dtype, shape=(dim_size, shared_size), min_value=2, max_value=5
        )
    )
    vec = draw(
        helpers.array_values(
            dtype=dtype, shape=(shared_size,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype, shape=(dim_size,), min_value=2, max_value=5
            )
        )
        return [dtype], input, mat, vec
    return [dtype], mat, vec


# addbmm
@handle_frontend_test(
    fn_tree="torch.addbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addbmm(
    *,
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        batch1=mat1,
        batch2=mat2,
        beta=beta,
        alpha=alpha,
    )


# addmm
@handle_frontend_test(
    fn_tree="torch.addmm",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addmm(
    *,
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        mat1=mat1,
        mat2=mat2,
        beta=beta,
        alpha=alpha,
    )


# addmv
@handle_frontend_test(
    fn_tree="torch.addmv",
    dtype_and_matrices=_get_dtype_input_and_mat_vec(with_input=True, skip_float16=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addmv(
    *,
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, mat, vec = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        input=input,
        mat=mat,
        vec=vec,
        beta=beta,
        alpha=alpha,
    )


# addr
@handle_frontend_test(
    fn_tree="torch.addr",
    dtype_and_vecs=_get_dtype_input_and_vectors(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addr(
    *,
    dtype_and_vecs,
    beta,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, vec1, vec2 = dtype_and_vecs

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        vec1=vec1,
        vec2=vec2,
        beta=beta,
        alpha=alpha,
    )


# baddbmm
@handle_frontend_test(
    fn_tree="torch.baddbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_baddbmm(
    *,
    dtype_and_matrices,
    beta,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, batch1, batch2 = dtype_and_matrices

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        input=input,
        batch1=batch1,
        batch2=batch2,
        beta=beta,
        alpha=alpha,
    )


# bmm
@handle_frontend_test(
    fn_tree="torch.bmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(),
)
def test_torch_bmm(
    *,
    dtype_and_matrices,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        input=mat1,
        mat2=mat2,
    )


# mm
@handle_frontend_test(
    fn_tree="torch.mm",
    dtype_xy=_get_dtype_input_and_matrices(),
)
def test_torch_mm(
    *,
    dtype_xy,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-02,
        rtol=1e-02,
        input=x,
        mat2=y,
    )


# mv
@handle_frontend_test(
    fn_tree="torch.mv",
    dtype_mat_vec=_get_dtype_input_and_mat_vec(skip_float16=True),
)
def test_torch_mv(
    *,
    dtype_mat_vec,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, mat, vec = dtype_mat_vec
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        input=mat,
        vec=vec,
        out=None,
    )
