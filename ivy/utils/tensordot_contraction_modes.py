import ivy


def _get_valid_contraction_modes_for_axes(shape1, shape2, axes):
    if isinstance(axes, int):
        modes1 = list(range(-axes, 0))
        modes2 = list(range(0, axes))
    else:
        modes1, modes2 = _make_tuple_of_2_elem(axes)
    modes1, modes2 = _convert_into_list(modes1, modes2)
    _check_equal_length(
        modes1,
        modes2,
        axes,
        error_message=(
            "Both tensors must have the same number of modes to contract along. "
        ),
    )

    _check_same_dimensions(
        modes1,
        modes2,
        shape1,
        shape2,
        error_message=(
            "Contraction dimensions must have the same dimensions in both tensors"
            " but got"
        ),
    )
    _make_negative_dims_positive(modes1, modes2, shape1, shape2)

    return modes1, modes2


def _get_valid_contraction_modes_for_batches(shape1, shape2, batched_modes):
    if isinstance(batched_modes, int):
        modes1 = [batched_modes]
        modes2 = [batched_modes]
    else:
        modes1, modes2 = _make_tuple_of_2_elem(batched_modes)
    modes1, modes2 = _convert_into_list(modes1, modes2)
    _check_equal_length(
        modes1,
        modes2,
        batched_modes,
        error_message="Both tensors must have the same number of batched modes",
    )
    _check_same_dimensions(
        modes1,
        modes2,
        shape1,
        shape2,
        error_message=(
            "Batch-dimensions must have the same dimensions in both tensors but got"
        ),
    )
    _make_negative_dims_positive(modes1, modes2, shape1, shape2)
    return modes1, modes2


def _check_equal_length(modes1, modes2, batched_modes, error_message):
    if len(modes1) != len(modes2):
        raise ValueError(
            error_message
            + f"However, got modes={batched_modes},  i.e. {len(modes1)} modes for"
            f" tensor 1 and {len(modes2)} mode for tensor 2(modes1={modes1}, and"
            f" modes2={modes2})"
        )


def _check_same_dimensions(modes1, modes2, shape1, shape2, error_message):
    for i in range(len(modes1)):
        if shape1[modes1[i]] != shape2[modes2[i]]:
            raise ValueError(
                error_message
                + f" mode {modes1[i]} of size {shape1[modes1[i]]} and "
                f" mode {modes2[i]} of size {shape2[modes2[i]]}."
            )


def _make_tuple_of_2_elem(modes):
    try:
        modes1, modes2 = modes
    except ValueError:
        modes1 = modes
        modes2 = modes
    return modes1, modes2


def _convert_into_list(modes1, modes2):
    try:
        modes1 = list(modes1)
    except TypeError:
        modes1 = [modes1]
    try:
        modes2 = list(modes2)
    except TypeError:
        modes2 = [modes2]
    return modes1, modes2


def _make_negative_dims_positive(modes1, modes2, shape1, shape2):
    ndim1 = len(shape1)
    ndim2 = len(shape2)
    for i in range(len(modes1)):
        if modes1[i] < 0:
            modes1[i] += ndim1
        if modes2[i] < 0:
            modes2[i] += ndim2


def _final_modes(x1, modes1, batch_modes1):
    final_modes = []
    n_batches = len(batch_modes1)
    batch_counter = 0
    free_counter = 0
    for i in range(ivy.get_num_dims(x1)):
        if i in modes1:
            continue
        elif i in batch_modes1:
            final_modes.append(batch_counter)
            batch_counter += 1
        else:
            final_modes.append(free_counter + n_batches)
            free_counter += 1
    return final_modes
