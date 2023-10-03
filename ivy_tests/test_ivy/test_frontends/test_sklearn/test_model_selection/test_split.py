from hypothesis import strategies as st

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test, handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.sklearn.model_selection"


@handle_frontend_method(
    class_tree=CLASS_TREE + ".KFold",
    init_tree="sklearn.model_selection.KFold",
    method_name="get_n_splits",
    dtype_x=helpers.dtype_and_values(),
)
def test_sklearn_kfold_get_n_split(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "n_splits": 2,  # todo test for shuffle
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "X": x[0],  # this arg only for compatibility
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE + ".KFold",
    init_tree="sklearn.model_selection.KFold",
    method_name="split",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_sklearn_kfold_split(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "n_splits": 2,  # todo test for shuffle
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "X": x[0],
            "y": x[1],
            "groups": None,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE + ".StratifiedKFold",
    init_tree="sklearn.model_selection.StratifiedKFold",
    method_name="split",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        shared_dtype=True,
        num_arrays=2,
        max_num_dims=2,
        min_num_dims=1,
    ),
)
def test_sklearn_stratfiedkfold_split(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "n_splits": 2,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "X": x[0],
            "y": x[1],
            "groups": None,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE + ".StratifiedKFold",
    init_tree="sklearn.model_selection.StratifiedKFold",
    method_name="get_n_splits",
    dtype_x=helpers.dtype_and_values(),
)
def test_sklearn_stratifiedkfold_get_n_split(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "n_splits": 2,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "X": x[0],  # for compatibility
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


@handle_frontend_test(
    fn_tree="sklearn.model_selection.train_test_split",
    arrays_and_dtypes=helpers.dtype_and_values(
        num_arrays=helpers.ints(min_value=2, max_value=4),
        shape=helpers.lists(
            x=helpers.ints(min_value=2, max_value=5),
            min_size=2,
            max_size=3,
        ),
    ),
    shuffle=st.booleans(),
)
def test_sklearn_test_train_split(
    arrays_and_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    shuffle,
):
    dtypes, values = arrays_and_dtypes
    kw = {}
    for i, x_ in enumerate(values):
        kw[f"x{i}"] = x_
    test_flags.num_positional_args = len(values)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
        **kw,
        shuffle=shuffle,
    )
