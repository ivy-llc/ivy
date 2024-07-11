import pytest
from hypothesis import strategies as st
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# --- Helpers --- #
# --------------- #


@st.composite
def _random_cp_data(draw):
    shape = draw(
        st.lists(helpers.ints(min_value=1, max_value=5), min_size=2, max_size=4)
    )
    rank = draw(helpers.ints(min_value=1, max_value=10))
    dtype = draw(helpers.get_dtypes("float", full=False))
    full = draw(st.booleans())
    orthogonal = draw(st.booleans())
    if (rank > min(shape)) and orthogonal:
        rank = min(shape)
    seed = draw(st.one_of((st.just(None), helpers.ints(min_value=0, max_value=2000))))
    normalise_factors = draw(st.booleans())
    return shape, rank, dtype[0], full, orthogonal, seed, normalise_factors


@st.composite
def _random_parafac2_data(draw):
    num_shapes = draw(st.integers(min_value=2, max_value=4))
    common_dimension = draw(st.integers(min_value=1, max_value=5))

    shapes = [
        (draw(st.integers(min_value=1, max_value=10)), common_dimension)
        for _ in range(num_shapes)
    ]

    rank = draw(helpers.ints(min_value=1, max_value=10))
    dtype = draw(helpers.get_dtypes("float", full=False))
    full = draw(st.booleans())
    seed = draw(st.one_of((st.just(None), helpers.ints(min_value=0, max_value=2000))))
    normalise_factors = draw(st.booleans())
    return shapes, rank, dtype[0], full, seed, normalise_factors


@st.composite
def _random_tr_data(draw):
    shape = draw(
        st.lists(helpers.ints(min_value=1, max_value=5), min_size=2, max_size=4)
    )
    rank = min(shape)
    dtype = draw(helpers.get_dtypes("valid", full=False))
    full = draw(st.booleans())
    seed = draw(st.one_of((st.just(None), helpers.ints(min_value=0, max_value=2000))))
    return shape, rank, dtype[0], full, seed


@st.composite
def _random_tt_data(draw):
    shape = draw(
        st.lists(helpers.ints(min_value=1, max_value=5), min_size=2, max_size=4)
    )
    rank = draw(helpers.ints(min_value=1, max_value=len(shape)))
    dtype = draw(helpers.get_dtypes("float", full=False))
    full = draw(st.booleans())
    seed = draw(st.one_of((st.just(None), helpers.ints(min_value=0, max_value=2000))))
    return shape, rank, dtype[0], full, seed


@st.composite
def _random_tucker_data(draw):
    shape = draw(
        st.lists(helpers.ints(min_value=1, max_value=5), min_size=2, max_size=4)
    )
    rank = []
    for dim in shape:
        rank.append(draw(helpers.ints(min_value=1, max_value=dim)))
    dtype = draw(helpers.get_dtypes("float", full=False))
    full = draw(st.booleans())
    orthogonal = draw(st.booleans())
    seed = draw(st.one_of((st.just(None), helpers.ints(min_value=0, max_value=2000))))
    non_negative = draw(st.booleans())
    return shape, rank, dtype[0], full, orthogonal, seed, non_negative


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


# --- Main --- #
# ------------ #


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


# mel_weight_matrix
@handle_test(
    fn_tree="functional.ivy.experimental.mel_weight_matrix",
    num_mel_bins=helpers.ints(min_value=5, max_value=10),
    dft_length=helpers.ints(min_value=5, max_value=10),
    sample_rate=helpers.ints(min_value=1000, max_value=2000),
    lower_edge_hertz=helpers.floats(min_value=0.0, max_value=5.0),
    upper_edge_hertz=helpers.floats(min_value=5.0, max_value=10.0),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_mel_weight_matrix(
    *,
    num_mel_bins,
    dft_length,
    sample_rate,
    lower_edge_hertz,
    upper_edge_hertz,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    helpers.test_function(
        input_dtypes=[],
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        rtol_=0.05,
        atol_=0.05,
        fn_name=fn_name,
        num_mel_bins=num_mel_bins,
        dft_length=dft_length,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
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
        assert index1 == index2
        assert x1 == x2.to_numpy()


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


# polyval
@handle_test(
    fn_tree="functional.ivy.experimental.polyval",
    dtype_and_coeffs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=0,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    test_instance_method=st.just(False),
)
def test_polyval(
    *, dtype_and_coeffs, dtype_and_x, test_flags, backend_fw, fn_name, on_device
):
    coeffs_dtype, coeffs = dtype_and_coeffs
    x_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=coeffs_dtype + x_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        coeffs=coeffs,
        x=x,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.random_cp",
    data=_random_cp_data(),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_random_cp(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    shape, rank, dtype, full, orthogonal, seed, normalise_factors = data
    results = helpers.test_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        fn_name=fn_name,
        shape=shape,
        rank=rank,
        dtype=dtype,
        full=full,
        orthogonal=orthogonal,
        seed=seed,
        normalise_factors=normalise_factors,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    if full:
        reconstructed_tensor = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
        reconstructed_tensor_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
        )
        for x, x_gt in zip(reconstructed_tensor, reconstructed_tensor_gt):
            assert np.prod(shape) == np.prod(x.shape)
            assert np.prod(shape) == np.prod(x_gt.shape)

    else:
        weights = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
        factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
        weights_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
        )
        factors_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
        )

        for w, w_gt in zip(weights, weights_gt):
            assert len(w) == rank
            assert len(w_gt) == rank

        for f, f_gt in zip(factors, factors_gt):
            assert np.prod(f.shape) == np.prod(f_gt.shape)


@handle_test(
    fn_tree="functional.ivy.experimental.random_tr",
    data=_random_tr_data(),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_random_tr(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    shape, rank, dtype, full, seed = data
    results = helpers.test_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        fn_name=fn_name,
        shape=shape,
        rank=rank,
        dtype=dtype,
        full=full,
        seed=seed,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    if full:
        reconstructed_tensor = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
        reconstructed_tensor_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
        )
        for x, x_gt in zip(reconstructed_tensor, reconstructed_tensor_gt):
            assert np.prod(shape) == np.prod(x.shape)
            assert np.prod(shape) == np.prod(x_gt.shape)

    else:
        core = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
        factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
        core_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
        )
        factors_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
        )

        for c, c_gt in zip(core, core_gt):
            assert len(c) == rank
            assert len(c_gt) == rank

        for f, f_gt in zip(factors, factors_gt):
            assert np.prod(f.shape) == np.prod(f_gt.shape)


def test_random_tr_throws_error_when_rank_first_last_elem_not_equal():
    rank = [2, 3]
    shape = [1, 2, 3]
    with pytest.raises(ValueError) as e:
        ivy.random_tr(shape, rank)
    assert e.value.args


# **Uncomment when Tensorly validation issue is resolved.**
# https://github.com/tensorly/tensorly/issues/528
# @handle_test(
#     fn_tree="functional.ivy.experimental.random_parafac2",
#     data=_random_parafac2_data(),
#     test_with_out=st.just(False),
#     test_instance_method=st.just(False),
# )
# def test_random_parafac2(
#     *,
#     data,
#     test_flags,
#     backend_fw,
#     fn_name,
#     on_device,
# ):
#     shapes, rank, dtype, full, seed, normalise_factors = data
#     results = helpers.test_function(
#         input_dtypes=[],
#         backend_to_test=backend_fw,
#         test_flags=test_flags,
#         on_device=on_device,
#         fn_name=fn_name,
#         shapes=shapes,
#         rank=rank,
#         dtype=dtype,
#         full=full,
#         seed=seed,
#         normalise_factors=normalise_factors,
#         test_values=False,
#     )
#     ret_np, ret_from_gt_np = results

#     if full:
#         reconstructed_tensor = helpers.flatten_and_to_np(ret=ret_np,
#                                                   backend=backend_fw)
#         reconstructed_tensor_gt = helpers.flatten_and_to_np(
#             ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
#         )

#         for x, x_gt in zip(reconstructed_tensor, reconstructed_tensor_gt):
#             assert x_gt.shape == x.shape

#     else:
#         weights = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
#         factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
#         # projections = helpers.flatten_and_to_np(ret=ret_np[2], backend=backend_fw)
#         weights_gt = helpers.flatten_and_to_np(
#             ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
#         )
#         factors_gt = helpers.flatten_and_to_np(
#             ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
#         )
#         # projections_gt = helpers.flatten_and_to_np(
#         #     ret=ret_from_gt_np[2], backend=test_flags.ground_truth_backend
#         # )

#         for w, w_gt in zip(weights, weights_gt):
#             assert len(w) == rank
#             assert len(w_gt) == rank

#         for f, f_gt in zip(factors, factors_gt):
#             assert np.prod(f.shape) == np.prod(f_gt.shape)

#         # for p, p_gt in zip(projections,projections_gt):
#         #     assert np.prod(p.shape) == np.prod(p_gt.shape)


@handle_test(
    fn_tree="functional.ivy.experimental.random_tt",
    data=_random_tt_data(),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
)
def test_random_tt(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    shape, rank, dtype, full, seed = data
    results = helpers.test_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        fn_name=fn_name,
        shape=shape,
        rank=rank,
        dtype=dtype,
        full=full,
        seed=seed,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    if full:
        reconstructed_tensor = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
        reconstructed_tensor_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
        )
        for x, x_gt in zip(reconstructed_tensor, reconstructed_tensor_gt):
            assert np.prod(shape) == np.prod(x.shape)
            assert np.prod(shape) == np.prod(x_gt.shape)

    else:
        factors = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
        factors_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
        )
        for f, f_gt in zip(factors, factors_gt):
            assert np.prod(f.shape) == np.prod(f_gt.shape)


@handle_test(
    fn_tree="functional.ivy.experimental.random_tucker",
    data=_random_tucker_data(),
    test_with_out=st.just(False),
    test_instance_method=st.just(False),
    test_gradients=st.just(False),
)
def test_random_tucker(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    shape, rank, dtype, full, orthogonal, seed, non_negative = data
    results = helpers.test_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        fn_name=fn_name,
        shape=shape,
        rank=rank,
        dtype=dtype,
        full=full,
        orthogonal=orthogonal,
        seed=seed,
        non_negative=non_negative,
        test_values=False,
    )

    ret_np, ret_from_gt_np = results

    if full:
        reconstructed_tensor = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
        reconstructed_tensor_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np, backend=test_flags.ground_truth_backend
        )
        for x, x_gt in zip(reconstructed_tensor, reconstructed_tensor_gt):
            assert np.prod(shape) == np.prod(x.shape)
            assert np.prod(shape) == np.prod(x_gt.shape)

    else:
        core = helpers.flatten_and_to_np(ret=ret_np[0], backend=backend_fw)
        factors = helpers.flatten_and_to_np(ret=ret_np[1], backend=backend_fw)
        core_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np[0], backend=test_flags.ground_truth_backend
        )
        factors_gt = helpers.flatten_and_to_np(
            ret=ret_from_gt_np[1], backend=test_flags.ground_truth_backend
        )

        for c, c_gt in zip(core, core_gt):
            assert np.prod(c.shape) == np.prod(rank)
            assert np.prod(c_gt.shape) == np.prod(rank)

        for f, f_gt in zip(factors, factors_gt):
            assert np.prod(f.shape) == np.prod(f_gt.shape)


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


@handle_test(
    fn_tree="functional.ivy.experimental.trilu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    upper=st.booleans(),
)
def test_trilu(*, dtype_and_x, k, upper, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        upper=upper,
        k=k,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.unsorted_segment_mean",
    d_x_n_s=valid_unsorted_segment_min_inputs(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_unsorted_segment_mean(
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
        backend_to_test=backend_fw,
        fn_name=fn_name,
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )


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
        backend_to_test=backend_fw,
        fn_name=fn_name,
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )


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
