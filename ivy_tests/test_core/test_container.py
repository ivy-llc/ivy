# global
import os
import pytest
import random
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


def test_container_from_dict(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([2])
    assert container.b.c == ivy.array([2])
    assert container['b']['d'] == ivy.array([3])
    assert container.b.d == ivy.array([3])


def test_container_expand_dims(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_expanded_dims = container.expand_dims(0)
    assert (container_expanded_dims['a'] == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims.a == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims['b']['c'] == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims.b.c == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_expanded_dims.b.d == ivy.array([[3]]))[0, 0]


def test_container_at_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    sub_container = container.at_key_chain('b')
    assert (sub_container['c'] == ivy.array([2]))[0]
    sub_container = container.at_key_chain('b/c')
    assert (sub_container == ivy.array([2]))[0]


def test_container_prune_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_pruned = container.prune_key_chain('b/c')
    assert (container_pruned['a'] == ivy.array([[1]]))[0, 0]
    assert (container_pruned.a == ivy.array([[1]]))[0, 0]
    assert (container_pruned['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_pruned.b.d == ivy.array([[3]]))[0, 0]
    assert ('c' not in container_pruned['b'].keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)

    container_further_pruned = container.prune_key_chain('b')
    assert (container_further_pruned['a'] == ivy.array([[1]]))[0, 0]
    assert (container_further_pruned.a == ivy.array([[1]]))[0, 0]
    assert ('b' not in container_further_pruned.keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b
            return False
        except AttributeError:
            return True

    assert _test_exception(container_further_pruned)


def test_container_prune_empty(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': {}, 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_pruned = container.prune_empty()
    assert (container_pruned['a'] == ivy.array([[1]]))[0, 0]
    assert (container_pruned.a == ivy.array([[1]]))[0, 0]
    assert (container_pruned['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_pruned.b.d == ivy.array([[3]]))[0, 0]
    assert ('c' not in container_pruned['b'].keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_shuffle(dev_str, call):
    if call is helpers.tf_graph_call:
        # tf.random.set_seed is not compiled. The shuffle is then not aligned between container items.
        pytest.skip()
    dict_in = {'a': ivy.array([1, 2, 3]),
               'b': {'c': ivy.array([1, 2, 3]), 'd': ivy.array([1, 2, 3])}}
    container = Container(dict_in)
    container_shuffled = container.shuffle(0)
    data = ivy.array([1, 2, 3])
    ivy.core.random.seed()
    shuffled_data = ivy.core.random.shuffle(data)

    assert np.array(container_shuffled['a'] == shuffled_data).all()
    assert np.array(container_shuffled.a == shuffled_data).all()
    assert np.array(container_shuffled['b']['c'] == shuffled_data).all()
    assert np.array(container_shuffled.b.c == shuffled_data).all()
    assert np.array(container_shuffled['b']['d'] == shuffled_data).all()
    assert np.array(container_shuffled.b.d == shuffled_data).all()


def test_container_to_iterator(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_iterator = container.to_iterator()
    for (key, value), expected_value in zip(container_iterator,
                                            [ivy.array([1]), ivy.array([2]), ivy.array([3])]):
        assert value == expected_value


def test_container_map(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_iterator = container.map(lambda x, _: x + 1).to_iterator()
    for (key, value), expected_value in zip(container_iterator,
                                            [ivy.array([2]), ivy.array([3]), ivy.array([4])]):
        assert call(lambda x: x, value) == call(lambda x: x, expected_value)


def test_container_to_random(dev_str, call):
    dict_in = {'a': ivy.array([1.]),
               'b': {'c': ivy.array([2.]), 'd': ivy.array([3.])}}
    container = Container(dict_in)
    random_container = container.to_random()
    for (key, value), orig_value in zip(random_container.to_iterator(),
                                        [ivy.array([2]), ivy.array([3]), ivy.array([4])]):
        assert call(ivy.shape, value) == call(ivy.shape, orig_value)


def test_container_dtype(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2.]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    dtype_container = container.dtype()
    for (key, value), expected_value in zip(dtype_container.to_iterator(),
                                            [ivy.array([1]).dtype,
                                             ivy.array([2.]).dtype,
                                             ivy.array([3]).dtype]):
        assert value == expected_value


def test_container_with_entries_as_lists(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2.]), 'd': 'some string'}}
    container = Container(dict_in)
    container_w_list_entries = container.with_entries_as_lists()
    for (key, value), expected_value in zip(container_w_list_entries.to_iterator(),
                                            [[1],
                                             [2.],
                                             'some string']):
        assert value == expected_value


def test_container_to_and_from_disk_as_hdf5(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.hdf5'
    dict_in_1 = {'a': ivy.array([np.float32(1.)]),
                 'b': {'c': ivy.array([np.float32(2.)]), 'd': ivy.array([np.float32(3.)])}}
    container1 = Container(dict_in_1)
    dict_in_2 = {'a': ivy.array([np.float32(1.), np.float32(1.)]),
                 'b': {'c': ivy.array([np.float32(2.), np.float32(2.)]),
                       'd': ivy.array([np.float32(3.), np.float32(3.)])}}
    container2 = Container(dict_in_2)

    # saving
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_hdf5(save_filepath, slice(1))
    assert np.array_equal(loaded_container.a, container1.a)
    assert np.array_equal(loaded_container.b.c, container1.b.c)
    assert np.array_equal(loaded_container.b.d, container1.b.d)

    # appending
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2, starting_index=1)
    assert os.path.exists(save_filepath)

    # loading after append
    loaded_container = Container.from_disk_as_hdf5(save_filepath)
    assert np.array_equal(loaded_container.a, container2.a)
    assert np.array_equal(loaded_container.b.c, container2.b.c)
    assert np.array_equal(loaded_container.b.d, container2.b.d)

    # load slice
    loaded_sliced_container = Container.from_disk_as_hdf5(save_filepath, slice(1, 2))
    assert np.array_equal(loaded_sliced_container.a, container1.a)
    assert np.array_equal(loaded_sliced_container.b.c, container1.b.c)
    assert np.array_equal(loaded_sliced_container.b.d, container1.b.d)

    # file size
    file_size, batch_size = Container.h5_file_size(save_filepath)
    assert file_size == 6 * np.dtype(np.float32).itemsize
    assert batch_size == 2

    os.remove(save_filepath)


def test_container_to_disk_shuffle_and_from_disk_as_hdf5(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.hdf5'
    dict_in = {'a': ivy.array([1, 2, 3]),
               'b': {'c': ivy.array([1, 2, 3]), 'd': ivy.array([1, 2, 3])}}
    container = Container(dict_in)

    # saving
    container.to_disk_as_hdf5(save_filepath, max_batch_size=3)
    assert os.path.exists(save_filepath)

    # shuffling
    Container.shuffle_h5_file(save_filepath)

    # loading
    container_shuffled = Container.from_disk_as_hdf5(save_filepath, slice(3))

    # testing
    data = np.array([1, 2, 3])
    random.seed(0)
    random.shuffle(data)

    assert (ivy.to_numpy(container_shuffled['a']) == data).all()
    assert (ivy.to_numpy(container_shuffled.a) == data).all()
    assert (ivy.to_numpy(container_shuffled['b']['c']) == data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == data).all()
    assert (ivy.to_numpy(container_shuffled['b']['d']) == data).all()
    assert (ivy.to_numpy(container_shuffled.b.d) == data).all()

    os.remove(save_filepath)


def test_container_to_and_from_disk_as_pickled(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.p'
    dict_in_1 = {'a': ivy.array([np.float32(1.)]),
                 'b': {'c': ivy.array([np.float32(2.)]), 'd': ivy.array([np.float32(3.)])}}
    container1 = Container(dict_in_1)
    dict_in_2 = {'a': ivy.array([np.float32(1.), np.float32(1.)]),
                 'b': {'c': ivy.array([np.float32(2.), np.float32(2.)]),
                       'd': ivy.array([np.float32(3.), np.float32(3.)])}}
    container2 = Container(dict_in_2)

    # saving
    container1.to_disk_as_pickled(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_pickled(save_filepath)
    assert np.array_equal(loaded_container.a, container1.a)
    assert np.array_equal(loaded_container.b.c, container1.b.c)
    assert np.array_equal(loaded_container.b.d, container1.b.d)

    os.remove(save_filepath)
