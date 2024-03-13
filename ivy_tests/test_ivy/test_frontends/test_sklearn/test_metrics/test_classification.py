from hypothesis import strategies as st
import torch
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import numpy as np


@handle_frontend_test(
    fn_tree="sklearn.metrics.accuracy_score",
    arrays_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        num_arrays=2,
        min_value=-2,
        max_value=2,
        shared_dtype=True,
        shape=(helpers.ints(min_value=2, max_value=5)),
    ),
    normalize=st.booleans(),
)
def test_sklearn_accuracy_score(
    arrays_and_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    normalize,
):
    dtypes, values = arrays_and_dtypes
    # sklearn accuracy_score does not support continuous values
    for i in range(2):
        if "float" in dtypes[i]:
            values[i] = np.floor(values[i])
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        y_true=values[0],
        y_pred=values[1],
        normalize=normalize,
        sample_weight=None,
    )


@handle_frontend_test(
    fn_tree="sklearn.metrics.f1_score",
    arrays_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        max_value=1,  # F1 score is for binary classification
        shared_dtype=True,
        shape=(helpers.ints(min_value=2, max_value=5)),
    ),
    sample_weight=st.lists(
        st.floats(min_value=0.1, max_value=1), min_size=2, max_size=5
    ),
)
def test_sklearn_f1_score(
    arrays_and_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    sample_weight,
):
    dtypes, values = arrays_and_dtypes
    # Ensure the values are binary by rounding and converting to int
    for i in range(2):
        values[i] = np.round(values[i]).astype(int)

    # Adjust sample_weight to have the correct length
    sample_weight = np.array(sample_weight).astype(float)
    if len(sample_weight) != len(values[0]):
        # If sample_weight is shorter, extend it with ones
        sample_weight = np.pad(
            sample_weight,
            (0, max(0, len(values[0]) - len(sample_weight))),
            "constant",
            constant_values=1.0,
        )
        # If sample_weight is longer, truncate it
        sample_weight = sample_weight[: len(values[0])]

    # Detach tensors if they require grad before converting to NumPy arrays
    if backend_fw == "torch":
        values = [
            value.detach().numpy()
            if isinstance(value, torch.Tensor) and value.requires_grad
            else value
            for value in values
        ]

    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        y_true=values[0],
        y_pred=values[1],
        sample_weight=sample_weight,
    )


@handle_frontend_test(
    fn_tree="sklearn.metrics.precision_score",
    arrays_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        max_value=1,  # Precision score is for binary classification
        shared_dtype=True,
        shape=(helpers.ints(min_value=2, max_value=5)),
    ),
    sample_weight=st.lists(
        st.floats(min_value=0.1, max_value=1), min_size=2, max_size=5
    ),
)
def test_sklearn_precision_score(
    arrays_and_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    sample_weight,
):
    dtypes, values = arrays_and_dtypes
    # Ensure the values are binary by rounding and converting to int
    for i in range(2):
        values[i] = np.round(values[i]).astype(int)

    # Adjust sample_weight to have the correct length
    sample_weight = np.array(sample_weight).astype(float)
    if len(sample_weight) != len(values[0]):
        # If sample_weight is shorter, extend it with ones
        sample_weight = np.pad(
            sample_weight,
            (0, max(0, len(values[0]) - len(sample_weight))),
            "constant",
            constant_values=1.0,
        )
        # If sample_weight is longer, truncate it
        sample_weight = sample_weight[: len(values[0])]

    # Detach tensors if they require grad before converting to NumPy arrays
    if backend_fw == "torch":
        values = [
            (
                value.detach().numpy()
                if isinstance(value, torch.Tensor) and value.requires_grad
                else value
            )
            for value in values
        ]

    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        y_true=values[0],
        y_pred=values[1],
        sample_weight=sample_weight,
    )


@handle_frontend_test(
    fn_tree="sklearn.metrics.recall_score",
    arrays_and_dtypes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=0,
        max_value=1,  # Recall score is for binary classification
        shared_dtype=True,
        shape=(helpers.ints(min_value=2, max_value=5)),
    ),
    sample_weight=st.lists(
        st.floats(min_value=0.1, max_value=1), min_size=2, max_size=5
    ),
)
def test_sklearn_recall_score(
    arrays_and_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    sample_weight,
):
    dtypes, values = arrays_and_dtypes
    # Ensure the values are binary by rounding and converting to int
    for i in range(2):
        values[i] = np.round(values[i]).astype(int)

    # Adjust sample_weight to have the correct length
    sample_weight = np.array(sample_weight).astype(float)
    if len(sample_weight) != len(values[0]):
        # If sample_weight is shorter, extend it with ones
        sample_weight = np.pad(
            sample_weight,
            (0, max(0, len(values[0]) - len(sample_weight))),
            "constant",
            constant_values=1.0,
        )
        # If sample_weight is longer, truncate it
        sample_weight = sample_weight[: len(values[0])]

    # Detach tensors if they require grad before converting to NumPy arrays
    if backend_fw == "torch":
        values = [
            (
                value.detach().numpy()
                if isinstance(value, torch.Tensor) and value.requires_grad
                else value
            )
            for value in values
        ]

    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        y_true=values[0],
        y_pred=values[1],
        sample_weight=sample_weight,
    )
