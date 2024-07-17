import tensorflow as tf

import itertools

from .tensorflow__helpers import tensorflow__deep_flatten
from .tensorflow__helpers import tensorflow__parse_ellipsis
from .tensorflow__helpers import tensorflow__parse_slice
from .tensorflow__helpers import tensorflow_arange
from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_astype_1
from .tensorflow__helpers import tensorflow_broadcast_arrays
from .tensorflow__helpers import tensorflow_diff
from .tensorflow__helpers import tensorflow_empty
from .tensorflow__helpers import tensorflow_expand_dims
from .tensorflow__helpers import tensorflow_get_item
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_is_bool_dtype
from .tensorflow__helpers import tensorflow_meshgrid
from .tensorflow__helpers import tensorflow_nonzero
from .tensorflow__helpers import tensorflow_reshape_1
from .tensorflow__helpers import tensorflow_set_item
from .tensorflow__helpers import tensorflow_shape
from .tensorflow__helpers import tensorflow_size_1
from .tensorflow__helpers import tensorflow_stack
from .tensorflow__helpers import tensorflow_where


def tensorflow__parse_query(query, x_shape, scatter=False):
    query = (query,) if not isinstance(query, tuple) else query
    ag__result_list_0 = []
    for q in query:
        res = tensorflow_asarray(q) if isinstance(q, (tuple, list, int)) else q
        ag__result_list_0.append(res)
    query = ag__result_list_0
    ag__result_list_1 = []
    for i, q in enumerate(query):
        if tensorflow_is_array(q):
            res = i
            ag__result_list_1.append(res)
    non_slice_q_idxs = ag__result_list_1
    to_front = (
        len(non_slice_q_idxs) > 1
        and any(tensorflow_diff(non_slice_q_idxs) != 1)
        and non_slice_q_idxs[-1] < len(x_shape)
    )
    ag__result_list_2 = []
    for i, q in enumerate(query):
        if q is None:
            res = i
            ag__result_list_2.append(res)
    new_axes = ag__result_list_2
    ag__result_list_3 = []
    for q in query:
        if q is not None:
            res = q
            ag__result_list_3.append(res)
    query = ag__result_list_3
    query = [Ellipsis] if query == [] else query
    ellipsis_inds = None
    if any(q is Ellipsis for q in query):
        query, ellipsis_inds = tensorflow__parse_ellipsis(query, len(x_shape))
    ag__result_list_4 = []
    for i, v in enumerate(query):
        if tensorflow_is_array(v):
            res = i
            ag__result_list_4.append(res)
    array_inds = ag__result_list_4
    if array_inds:
        array_queries = tensorflow_broadcast_arrays(
            *[v for i, v in enumerate(query) if i in array_inds]
        )
        array_queries = [
            (
                tensorflow_nonzero(q, as_tuple=False)[0]
                if tensorflow_is_bool_dtype(q)
                else q
            )
            for q in array_queries
        ]
        array_queries = [
            (
                tensorflow_astype_1(
                    tensorflow_where(
                        arr < 0, arr + tensorflow_get_item(x_shape, i), arr
                    ),
                    tf.int64,
                )
                if tensorflow_size_1(arr)
                else tensorflow_astype_1(arr, tf.int64)
            )
            for arr, i in zip(array_queries, array_inds)
        ]
        for idx, arr in zip(array_inds, array_queries):
            query = tensorflow_set_item(query, idx, arr)
    ag__result_list_5 = []
    for i, q in enumerate(query):
        res = (
            tensorflow_astype_1(
                tensorflow__parse_slice(q, tensorflow_get_item(x_shape, i)), tf.int64
            )
            if isinstance(q, slice)
            else q
        )
        ag__result_list_5.append(res)
    query = ag__result_list_5
    if len(query) < len(x_shape):
        query = query + [
            tensorflow_astype_1(tensorflow_arange(0, s, 1), tf.int64)
            for s in tensorflow_get_item(x_shape, slice(len(query), None, None))
        ]
    if len(array_inds) and to_front:
        target_shape = (
            [list(array_queries[0].shape)]
            + [
                list(tensorflow_get_item(query, i).shape)
                for i in range(len(query))
                if i not in array_inds
            ]
            + [[] for _ in range(len(array_inds) - 1)]
        )
    elif len(array_inds):
        target_shape = (
            [list(tensorflow_get_item(query, i).shape) for i in range(0, array_inds[0])]
            + [list(tensorflow_shape(array_queries[0], as_array=True))]
            + [[] for _ in range(len(array_inds) - 1)]
            + [
                list(tensorflow_shape(tensorflow_get_item(query, i), as_array=True))
                for i in range(array_inds[-1] + 1, len(query))
            ]
        )
    else:
        target_shape = [list(q.shape) for q in query]
    if ellipsis_inds is not None:
        target_shape = (
            tensorflow_get_item(target_shape, slice(None, ellipsis_inds[0], None))
            + [
                tensorflow_get_item(
                    target_shape, slice(ellipsis_inds[0], ellipsis_inds[1], None)
                )
            ]
            + tensorflow_get_item(target_shape, slice(ellipsis_inds[1], None, None))
        )
    for i, ax in enumerate(new_axes):
        if len(array_inds) and to_front:
            ax = ax - (sum(1 for x in array_inds if x < ax) - 1)
            ax = ax + i
        target_shape = [
            *tensorflow_get_item(target_shape, slice(None, ax, None)),
            1,
            *tensorflow_get_item(target_shape, slice(ax, None, None)),
        ]
    target_shape = tensorflow__deep_flatten(target_shape)
    ag__result_list_6 = []
    for q in query:
        res = tensorflow_expand_dims(q) if not len(q.shape) else q
        ag__result_list_6.append(res)
    query = ag__result_list_6
    if len(array_inds):
        array_queries = [
            (
                tensorflow_reshape_1(arr, (-1,))
                if len(arr.shape) > 1
                else tensorflow_expand_dims(arr)
                if not len(arr.shape)
                else arr
            )
            for arr in array_queries
        ]
        array_queries = tensorflow_stack(array_queries, axis=1)
    if len(array_inds) == len(query):
        indices = tensorflow_reshape_1(array_queries, (*target_shape, len(x_shape)))
    elif len(array_inds) == 0:
        indices = tensorflow_reshape_1(
            tensorflow_stack(tensorflow_meshgrid(*query, indexing="ij"), axis=-1),
            (*target_shape, len(x_shape)),
        )
    elif to_front:
        post_array_queries = (
            tensorflow_reshape_1(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i not in array_inds],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, len(query) - len(array_inds)),
            )
            if len(array_inds) < len(query)
            else tensorflow_empty((1, 0))
        )
        indices = tensorflow_reshape_1(
            tensorflow_asarray(
                [
                    (*arr, *post)
                    for arr, post in itertools.product(
                        array_queries, post_array_queries
                    )
                ]
            ),
            (*target_shape, len(x_shape)),
        )
    else:
        pre_array_queries = (
            tensorflow_reshape_1(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i < array_inds[0]],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, array_inds[0]),
            )
            if array_inds[0] > 0
            else tensorflow_empty((1, 0))
        )
        post_array_queries = (
            tensorflow_reshape_1(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i > array_inds[-1]],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, len(query) - 1 - array_inds[-1]),
            )
            if array_inds[-1] < len(query) - 1
            else tensorflow_empty((1, 0))
        )
        indices = tensorflow_reshape_1(
            tensorflow_asarray(
                [
                    (*pre, *arr, *post)
                    for pre, arr, post in itertools.product(
                        pre_array_queries, array_queries, post_array_queries
                    )
                ]
            ),
            (*target_shape, len(x_shape)),
        )
    return (
        tensorflow_astype_1(indices, tf.int64),
        target_shape,
        array_inds if len(array_inds) and to_front else None,
    )
