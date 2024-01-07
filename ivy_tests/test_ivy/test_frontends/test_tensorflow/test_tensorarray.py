# global
from hypothesis import strategies as st, given
import numpy as np
import tensorflow as tf

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import BackendHandler


# --- Helpers --- #
# --------------- #


def _helper_init_tensorarray(backend_fw, l_kwargs, fn=None):
    id_write, kwargs = l_kwargs
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        local_importer = ivy_backend.utils.dynamic_import
        tf_frontend = local_importer.import_module(
            "ivy.functional.frontends.tensorflow"
        )
        ta = tf_frontend.tensor.TensorArray(**kwargs)
        ta_gt = tf.TensorArray(**kwargs)
        if fn == "unstack":
            ta_gt = ta_gt.unstack(tf.constant(id_write))
            ta = ta.unstack(tf_frontend.constant(id_write))
        elif fn == "split":
            ta_gt = ta_gt.split(**id_write)
            ta = ta.split(**id_write)
        elif fn == "scatter":
            indices, value = [*zip(*id_write)]
            ta_gt = ta_gt.scatter(indices, tf.cast(tf.stack(value), dtype=ta_gt.dtype))
            value = tf_frontend.stack(list(map(tf_frontend.constant, value)))
            ta = ta.scatter(indices, tf_frontend.cast(value, ta.dtype))
        else:
            for id, write in id_write:
                ta_gt = ta_gt.write(id, tf.constant(write))
                ta = ta.write(id, tf_frontend.constant(write))
    return ta_gt, ta


@st.composite
def _helper_random_tensorarray(draw, fn=None):
    size = draw(st.integers(1, 10))
    dynamic_size = draw(st.booleans())
    clear_after_read = draw(st.booleans())
    infer_shape = draw(st.booleans())
    element_shape = draw(helpers.get_shape())
    element_shape = draw(st.one_of(st.just(None), st.just(element_shape)))
    shape = None
    if (
        infer_shape
        or element_shape is not None
        or fn in ["scatter", "stack", "gather", "concat"]
    ):
        if fn == "concat":
            element_shape = None
            infer_shape = False
            shape = list(draw(helpers.get_shape(min_num_dims=1)))
        elif element_shape is None:
            shape = draw(helpers.get_shape())
        else:
            shape = element_shape
    dtype = draw(helpers.get_dtypes(full=False, prune_function=False))[0]
    if fn in ["stack", "concat"]:
        ids_to_write = [True for i in range(size)]
    else:
        ids_to_write = [draw(st.booleans()) for i in range(size)]
        if sum(ids_to_write) == 0:
            ids_to_write[draw(st.integers(0, size - 1))] = True
    kwargs = {
        "dtype": dtype,
        "size": size,
        "dynamic_size": dynamic_size,
        "clear_after_read": clear_after_read,
        "infer_shape": infer_shape,
        "element_shape": element_shape,
    }
    id_write = []
    for id, flag in enumerate(ids_to_write):
        if fn == "concat":
            shape[0] = draw(st.integers(1, 10))
        if flag:
            write = np.array(
                draw(
                    helpers.array_values(
                        dtype=dtype,
                        shape=shape if shape is not None else helpers.get_shape(),
                    )
                )
            )
            id_write.append((id, write))
    if fn != "gather":
        return id_write, kwargs
    else:
        ids = []
        for id, _ in id_write:
            if draw(st.booleans()):
                ids.append(id)
        if not ids:
            ids.append(id)
        return id_write, kwargs, ids


@st.composite
def _helper_split(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))
    dtype = draw(helpers.get_dtypes(full=False, prune_function=False))[0]
    value = draw(helpers.array_values(dtype=dtype, shape=shape))
    dynamic_size = draw(st.booleans())
    if dynamic_size:
        size = draw(st.integers(1, shape[0] + 5))
    else:
        size = shape[0]
    total = 0
    length = []
    for i in range(shape[0]):
        length.append(draw(st.integers(0, shape[0] - total)))
        total += length[-1]
    if total != shape[0]:
        length[-1] += shape[0] - total
    return {"value": value, "lengths": length}, {
        "dtype": dtype,
        "size": size,
        "dynamic_size": dynamic_size,
    }


@st.composite
def _helper_unstack(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))
    size = draw(st.integers(1, 10))
    dynamic_size = draw(st.booleans()) if size >= shape[0] else True
    dtype = draw(helpers.get_dtypes(full=False, prune_function=False))[0]
    tensor = draw(helpers.array_values(dtype=dtype, shape=shape))
    kwargs = {"dtype": dtype, "size": size, "dynamic_size": dynamic_size}
    return tensor, kwargs


# --- Main --- #
# ------------ #


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_close(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    ta.close()
    ta_gt.close()
    assert np.array(ta.size()) == 0
    assert np.array(ta_gt.size()) == 0


@given(l_kwargs=_helper_random_tensorarray(fn="concat"))
def test_tensorflow_concat(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    helpers.value_test(
        ret_np_from_gt_flat=ta_gt.concat().numpy().flatten(),
        ret_np_flat=np.array(ta.concat()).flatten(),
        backend=backend_fw,
    )


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_dtype(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    assert ta_gt.dtype == ta.dtype.ivy_dtype


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_dynamic_size(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    assert ta_gt.dynamic_size == ta.dynamic_size


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_element_shape(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    assert ta_gt.element_shape == ta.element_shape


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_flow(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    assert ta_gt.flow == ta.flow


@given(l_kwargs=_helper_random_tensorarray(fn="gather"))
def test_tensorflow_gather(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs[:2])
    *_, indices = l_kwargs
    helpers.value_test(
        ret_np_from_gt_flat=ta_gt.gather(indices).numpy().flatten(),
        ret_np_flat=np.array(ta.gather(indices)).flatten(),
        backend=backend_fw,
    )


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_handle(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    assert ta_gt.handle == ta.handle


# test for read and write methods
@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_read(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    id_read, _ = l_kwargs
    for id, read in id_read:
        helpers.value_test(
            ret_np_from_gt_flat=ta_gt.read(id).numpy().flatten(),
            ret_np_flat=np.array(ta.read(id)).flatten(),
            backend=backend_fw,
        )


@given(l_kwargs=_helper_random_tensorarray(fn="scatter"))
def test_tensorflow_scatter(
    l_kwargs,
    backend_fw,
):
    id_read, _ = l_kwargs
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs, "scatter")
    for id, read in id_read:
        helpers.value_test(
            ret_np_from_gt_flat=ta_gt.read(id).numpy().flatten(),
            ret_np_flat=np.array(ta.read(id)).flatten(),
            backend=backend_fw,
        )


@given(l_kwargs=_helper_random_tensorarray())
def test_tensorflow_size(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    helpers.value_test(
        ret_np_from_gt_flat=ta_gt.size().numpy().flatten(),
        ret_np_flat=np.array(ta.size()).flatten(),
        backend=backend_fw,
    )


@given(
    kwargs_v_l=_helper_split(),
)
def test_tensorflow_split(kwargs_v_l, backend_fw):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, kwargs_v_l, "split")
    for id in range(ta_gt.size()):
        helpers.value_test(
            ret_np_from_gt_flat=ta_gt.read(id).numpy().flatten(),
            ret_np_flat=np.array(ta.read(id)).flatten(),
            backend=backend_fw,
        )


@given(l_kwargs=_helper_random_tensorarray(fn="stack"))
def test_tensorflow_stack(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs)
    helpers.value_test(
        ret_np_from_gt_flat=ta_gt.stack().numpy().flatten(),
        ret_np_flat=np.array(ta.stack()).flatten(),
        backend=backend_fw,
    )


@given(l_kwargs=_helper_unstack())
def test_tensorflow_unstack(
    l_kwargs,
    backend_fw,
):
    ta_gt, ta = _helper_init_tensorarray(backend_fw, l_kwargs, "unstack")
    helpers.value_test(
        ret_np_from_gt_flat=ta_gt.stack().numpy().flatten(),
        ret_np_flat=np.array(ta.stack()).flatten(),
        backend=backend_fw,
    )
