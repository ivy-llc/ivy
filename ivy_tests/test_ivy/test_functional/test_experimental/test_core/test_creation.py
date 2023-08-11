from hypothesis import strategies as st
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# vorbis_window
@handle_test(
    fn_tree="functional.ivy.experimental.vorbis_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=1,
        max_value=10,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_vorbis_window(
    *, dtype_and_x, dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        atol_=1e-02,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        window_length=int(x[0]),
        dtype=dtype[0],
    )


# TODO: fix return precision problem when dtype=bfloat16
# hann_window
@handle_test(
    fn_tree="functional.ivy.experimental.hann_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_hann_window(
    *, dtype_and_x, periodic, dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        atol_=0.015,
        fn_name=fn_name,
        on_device=on_device,
        size=int(x[0]),
        periodic=periodic,
        dtype=dtype[0],
    )


# kaiser_window
@handle_test(
    fn_tree="functional.ivy.experimental.kaiser_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    beta=st.floats(min_value=0, max_value=5),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_kaiser_window(
    *, dtype_and_x, periodic, beta, dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        window_length=int(x[0]),
        periodic=periodic,
        beta=beta,
        dtype=dtype[0],
    )


# kaiser_bessel_derived_window
@handle_test(
    fn_tree="functional.ivy.experimental.kaiser_bessel_derived_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=1,
        max_value=10,
    ),
    beta=st.floats(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_kaiser_bessel_derived_window(
    *, dtype_and_x, beta, dtype, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        window_length=int(x[0]),
        beta=beta,
        dtype=dtype[0],
    )


# hamming_window
@handle_test(
    fn_tree="functional.ivy.experimental.hamming_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    dtype_and_f=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_num_dims=0,
        num_arrays=2,
        min_value=0,
        max_value=5,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_hamming_window(
    *,
    dtype_and_x,
    periodic,
    dtype_and_f,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype1, x = dtype_and_x
    input_dtype2, f = dtype_and_f
    helpers.test_function(
        input_dtypes=input_dtype1 + input_dtype2,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        atol_=2e-06,
        fn_name=fn_name,
        on_device=on_device,
        window_length=int(x[0]),
        periodic=periodic,
        alpha=float(f[0]),
        beta=float(f[1]),
        dtype=dtype[0],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.tril_indices",
    dtype_and_n=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        num_arrays=2,
        min_value=0,
        max_value=10,
    ),
    k=helpers.ints(min_value=-11, max_value=11),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_tril_indices(*, dtype_and_n, k, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_n
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        n_rows=int(x[0]),
        n_cols=int(x[1]),
        k=k,
    )


# eye_like
@handle_test(
    fn_tree="functional.ivy.experimental.eye_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_gradients=st.just(False),
    number_positional_args=st.just(1),
)
def test_eye_like(*, dtype_and_x, k, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        k=k,
        dtype=dtype[0],
        device=on_device,
    )


# ndenumerate
@handle_test(
    fn_tree="functional.ivy.experimental.ndenumerate",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
)
def test_ndenumerate(dtype_and_x):
    values = dtype_and_x[1][0]
    for (index1, x1), (index2, x2) in zip(
        np.ndenumerate(values), ivy.ndenumerate(values)
    ):
        assert index1 == index2 and x1 == x2.to_numpy()


# ndindex
@handle_test(
    fn_tree="functional.ivy.experimental.ndindex",
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        ret_shape=True,
    ),
)
def test_ndindex(dtype_x_shape):
    shape = dtype_x_shape[2]
    for index1, index2 in zip(np.ndindex(shape), ivy.ndindex(shape)):
        assert index1 == index2


# indices
@handle_test(
    fn_tree="functional.ivy.experimental.indices",
    ground_truth_backend="numpy",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=helpers.get_dtypes(
        "numeric",
        full=False,
    ),
    sparse=st.booleans(),
    container_flags=st.just([False]),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_indices(*, shape, dtypes, sparse, test_flags, backend_fw, fn_name, on_device):
    helpers.test_function(
        input_dtypes=[],
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        dimensions=shape,
        dtype=dtypes[0],
        sparse=sparse,
    )


@st.composite
def valid_unsorted_segment_min_inputs(draw):
    while True:
        dtype = draw(st.sampled_from([ivy.int32, ivy.int64, ivy.float32, ivy.float64]))
        segment_ids_dim = draw(st.integers(min_value=3, max_value=10))
        num_segments = draw(st.integers(min_value=2, max_value=segment_ids_dim))

        data_dim = draw(
            helpers.get_shape(
                min_dim_size=segment_ids_dim,
                max_dim_size=segment_ids_dim,
                min_num_dims=1,
                max_num_dims=4,
            )
        )
        data_dim = (segment_ids_dim,) + data_dim[1:]

        data = draw(
            helpers.array_values(
                dtype=dtype,
                shape=data_dim,
                min_value=1,
                max_value=10,
            )
        )

        segment_ids = draw(
            helpers.array_values(
                dtype=ivy.int32,
                shape=(segment_ids_dim,),
                min_value=0,
                max_value=num_segments + 1,
            )
        )
        if data.shape[0] == segment_ids.shape[0]:
            if np.max(segment_ids) < num_segments:
                return (dtype, ivy.int32), data, num_segments, segment_ids


# unsorted_segment_min
@handle_test(
    fn_tree="functional.ivy.experimental.unsorted_segment_min",
    d_x_n_s=valid_unsorted_segment_min_inputs(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_unsorted_segment_min(
    *,
    d_x_n_s,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, data, num_segments, segment_ids = d_x_n_s
    helpers.test_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        fn_name=fn_name,
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.unsorted_segment_sum",
    d_x_n_s=valid_unsorted_segment_min_inputs(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_unsorted_segment_sum(
    *,
    d_x_n_s,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, data, num_segments, segment_ids = d_x_n_s
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.complex",
    dtype_x_and_y=helpers.dtype_and_values(
        available_dtypes="valid",
        num_arrays=2,
        min_num_dims=1,
    ),
)
def test_complex(
    *,
    dtype_x_and_y,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, x = dtype_x_and_y
    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        real=x[0],
        imag=x[1],
    )
