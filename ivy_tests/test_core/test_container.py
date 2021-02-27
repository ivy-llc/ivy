# global
import os
import random
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container
from ivy_tests.helpers import mx_sym_to_val as func


def test_container_from_dict():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        assert fn(container['a']) == fn(ivy.array([1], f=lib))
        assert fn(container.a) == fn(ivy.array([1], f=lib))
        assert fn(container['b']['c']) == fn(ivy.array([2], f=lib))
        assert fn(container.b.c) == fn(ivy.array([2], f=lib))
        assert fn(container['b']['d']) == fn(ivy.array([3], f=lib))
        assert fn(container.b.d) == fn(ivy.array([3], f=lib))


def test_container_expand_dims():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        container_expanded_dims = container.expand_dims(0)
        assert (fn(container_expanded_dims['a']) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_expanded_dims.a) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_expanded_dims['b']['c']) == fn(ivy.array([[2]], f=lib)))[0, 0]
        assert (fn(container_expanded_dims.b.c) == fn(ivy.array([[2]], f=lib)))[0, 0]
        assert (fn(container_expanded_dims['b']['d']) == fn(ivy.array([[3]], f=lib)))[0, 0]
        assert (fn(container_expanded_dims.b.d) == fn(ivy.array([[3]], f=lib)))[0, 0]


def test_container_at_key_chain():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        sub_container = container.at_key_chain('b')
        assert (fn(sub_container['c']) == fn(ivy.array([2], f=lib)))[0]
        sub_container = container.at_key_chain('b/c')
        assert (fn(sub_container) == fn(ivy.array([2], f=lib)))[0]


def test_container_prune_key_chain():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        container_pruned = container.prune_key_chain('b/c')
        assert (fn(container_pruned['a']) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_pruned.a) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_pruned['b']['d']) == fn(ivy.array([[3]], f=lib)))[0, 0]
        assert (fn(container_pruned.b.d) == fn(ivy.array([[3]], f=lib)))[0, 0]
        assert ('c' not in container_pruned['b'].keys())

        def _test_exception(container_in):
            try:
                _ = container_in.b.c
                return False
            except AttributeError:
                return True

        assert _test_exception(container_pruned)

        container_further_pruned = container.prune_key_chain('b')
        assert (fn(container_further_pruned['a']) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_further_pruned.a) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert ('b' not in container_further_pruned.keys())

        def _test_exception(container_in):
            try:
                _ = container_in.b
                return False
            except AttributeError:
                return True

        assert _test_exception(container_further_pruned)


def test_container_prune_empty():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': {}, 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        container_pruned = container.prune_empty()
        assert (fn(container_pruned['a']) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_pruned.a) == fn(ivy.array([[1]], f=lib)))[0, 0]
        assert (fn(container_pruned['b']['d']) == fn(ivy.array([[3]], f=lib)))[0, 0]
        assert (fn(container_pruned.b.d) == fn(ivy.array([[3]], f=lib)))[0, 0]
        assert ('c' not in container_pruned['b'].keys())

        def _test_exception(container_in):
            try:
                _ = container_in.b.c
                return False
            except AttributeError:
                return True

        assert _test_exception(container_pruned)


def test_container_shuffle():
    for lib, call in helpers.calls():
        if call is helpers.tf_graph_call:
            # tf.random.set_seed is not compiled. The shuffle is then not aligned between container items.
            continue
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1, 2, 3], f=lib),
                   'b': {'c': ivy.array([1, 2, 3], f=lib), 'd': ivy.array([1, 2, 3], f=lib)}}
        container = Container(dict_in)
        container_shuffled = container.shuffle(0)
        data = ivy.array([1, 2, 3], f=lib)
        ivy.core.random.seed(f=lib)
        shuffled_data = ivy.core.random.shuffle(data)

        assert np.array(fn(container_shuffled['a']) == fn(shuffled_data)).all()
        assert np.array(fn(container_shuffled.a) == fn(shuffled_data)).all()
        assert np.array(fn(container_shuffled['b']['c']) == fn(shuffled_data)).all()
        assert np.array(fn(container_shuffled.b.c) == fn(shuffled_data)).all()
        assert np.array(fn(container_shuffled['b']['d']) == fn(shuffled_data)).all()
        assert np.array(fn(container_shuffled.b.d) == fn(shuffled_data)).all()


def test_container_to_iterator():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            fn = func
        else:
            fn = lambda x: x
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        container_iterator = container.to_iterator()
        for (key, value), expected_value in zip(container_iterator,
                                                [ivy.array([1], f=lib), ivy.array([2], f=lib), ivy.array([3], f=lib)]):
            assert fn(value) == fn(expected_value)


def test_container_map():
    for lib, call in helpers.calls():
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        container_iterator = container.map(lambda x, _: x + 1).to_iterator()
        for (key, value), expected_value in zip(container_iterator,
                                                [ivy.array([2], f=lib), ivy.array([3], f=lib), ivy.array([4], f=lib)]):
            assert call(lambda x: x, value) == call(lambda x: x, expected_value)


def test_container_to_random():
    for lib, call in helpers.calls():
        dict_in = {'a': ivy.array([1.], f=lib),
                   'b': {'c': ivy.array([2.], f=lib), 'd': ivy.array([3.], f=lib)}}
        container = Container(dict_in)
        random_container = container.to_random(lib)
        for (key, value), orig_value in zip(random_container.to_iterator(),
                                            [ivy.array([2], f=lib), ivy.array([3], f=lib), ivy.array([4], f=lib)]):
            assert call(ivy.shape, value, f=lib) == call(ivy.shape, orig_value, f=lib)


def test_container_dtype():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # MXNet symbolic does not support dtype
            continue
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2.], f=lib), 'd': ivy.array([3], f=lib)}}
        container = Container(dict_in)
        dtype_container = container.dtype()
        for (key, value), expected_value in zip(dtype_container.to_iterator(),
                                                [ivy.array([1], f=lib).dtype,
                                                 ivy.array([2.], f=lib).dtype,
                                                 ivy.array([3], f=lib).dtype]):
            assert value == expected_value


def test_container_with_entries_as_lists():
    for lib, call in helpers.calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # to_list() requires eager execution
            continue
        dict_in = {'a': ivy.array([1], f=lib),
                   'b': {'c': ivy.array([2.], f=lib), 'd': 'some string'}}
        container = Container(dict_in)
        container_w_list_entries = container.with_entries_as_lists(lib)
        for (key, value), expected_value in zip(container_w_list_entries.to_iterator(),
                                                [[1],
                                                 [2.],
                                                 'some string']):
            assert value == expected_value


def test_container_to_and_from_disk():
    for lib, call in helpers.calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # container disk saving requires eager execution
            continue
        save_filepath = 'container_on_disk.hdf5'
        dict_in_1 = {'a': ivy.array([np.float32(1.)], f=lib),
                     'b': {'c': ivy.array([np.float32(2.)], f=lib), 'd': ivy.array([np.float32(3.)], f=lib)}}
        container1 = Container(dict_in_1)
        dict_in_2 = {'a': ivy.array([np.float32(1.), np.float32(1.)], f=lib),
                     'b': {'c': ivy.array([np.float32(2.), np.float32(2.)], f=lib),
                           'd': ivy.array([np.float32(3.), np.float32(3.)], f=lib)}}
        container2 = Container(dict_in_2)

        # saving
        container1.to_disk(save_filepath, max_batch_size=2)
        assert os.path.exists(save_filepath)

        # loading
        loaded_container = Container.from_disk(save_filepath, lib, slice(1))
        assert np.array_equal(loaded_container.a, container1.a)
        assert np.array_equal(loaded_container.b.c, container1.b.c)
        assert np.array_equal(loaded_container.b.d, container1.b.d)

        # appending
        container1.to_disk(save_filepath, max_batch_size=2, starting_index=1)
        assert os.path.exists(save_filepath)

        # loading after append
        loaded_container = Container.from_disk(save_filepath, lib)
        assert np.array_equal(loaded_container.a, container2.a)
        assert np.array_equal(loaded_container.b.c, container2.b.c)
        assert np.array_equal(loaded_container.b.d, container2.b.d)

        # load slice
        loaded_sliced_container = Container.from_disk(save_filepath, lib, slice(1, 2))
        assert np.array_equal(loaded_sliced_container.a, container1.a)
        assert np.array_equal(loaded_sliced_container.b.c, container1.b.c)
        assert np.array_equal(loaded_sliced_container.b.d, container1.b.d)

        # file size
        file_size, batch_size = Container.h5_file_size(save_filepath)
        assert file_size == 6 * np.dtype(np.float32).itemsize
        assert batch_size == 2

        os.remove(save_filepath)


def test_container_to_disk_shuffle_and_from_disk():
    for lib, call in helpers.calls():
        if call in [helpers.tf_graph_call, helpers.mx_graph_call]:
            # container disk saving requires eager execution
            continue
        save_filepath = 'container_on_disk.hdf5'
        dict_in = {'a': ivy.array([1, 2, 3], f=lib),
                   'b': {'c': ivy.array([1, 2, 3], f=lib), 'd': ivy.array([1, 2, 3], f=lib)}}
        container = Container(dict_in)

        # saving
        container.to_disk(save_filepath, max_batch_size=3)
        assert os.path.exists(save_filepath)

        # shuffling
        Container.shuffle_h5_file(save_filepath)

        # loading
        container_shuffled = Container.from_disk(save_filepath, lib, slice(3))

        # testing
        data = np.array([1, 2, 3])
        random.seed(0)
        random.shuffle(data)

        assert (ivy.to_numpy(container_shuffled['a'], lib) == data).all()
        assert (ivy.to_numpy(container_shuffled.a, lib) == data).all()
        assert (ivy.to_numpy(container_shuffled['b']['c'], lib) == data).all()
        assert (ivy.to_numpy(container_shuffled.b.c, lib) == data).all()
        assert (ivy.to_numpy(container_shuffled['b']['d'], lib) == data).all()
        assert (ivy.to_numpy(container_shuffled.b.d, lib) == data).all()

        os.remove(save_filepath)
