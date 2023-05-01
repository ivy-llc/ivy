from hypothesis import strategies as st
import hypothesis.extra.numpy as nph
# local
import numpy as np
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
import tensorflow as tf
import ivy.functional.backends.numpy.experimental.manipulation as npp
import ivy.functional.backends.tensorflow.experimental.manipulation as tnp


def foo():
    dtype_and_input_and_other = (['uint32'],
                                 ivy.array([0]),
                                 ((0, 1),),
                                 ((2, 2),),
                                 ((0, 1),),
                                 ((0, 0),),
                                 'constant')
    reflect_type = 'even'
    fn_name = 'pad'
    ground_truth_backend = 'numpy'
    num_positional_args = 2.
    with_out = False
    instance_method = False
    native_arrays = [False]
    container = [False]
    as_variable = [False]
    test_gradients = False
    test_compile = None
    fw = 'ivy.functional.backends.tensorflow'
    on_device = 'cpu'

    helpers.test_function(
    input_dtypes = ['uint32'],
    as_variable_flags = [False],
    with_out = False,
    num_positional_args = 2,
    native_array_flags = [False],
    container_flags = [False],
    instance_method = False,
    test_gradients = False,
    test_compile = None,
    fw = 'ivy.functional.backends.tensorflow',
    on_device = 'cpu',
    fn_name = 'pad',
    ground_truth_backend='numpy',
    x = ivy.array([0]),
    pad_width = ((0,0),),
    mode = 'constant'
    )



foo()