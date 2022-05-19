# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


def test_docstrings():
    failures = list()
    success = True
    
    ''' 
        Functions that are skipped due to precision problem include:
            add, cross_entropy, std

        Functions that are skipped due to output errors:
            roll, array_equal, cross_entropy, trace, shape

        Functions that are skipped due to tensorflow returning float instead of int:
            full_like, ones_like, zeros_like, bitwise_invert, 
            copy_array, expand_dims, reshape, einsum

        Functions skipped due to some runtime error: conv3d, tan

        Functions skipped as their output dependent on outside factors:
            random_normal, random_uniform, shuffle, num_gpus
    '''
    skip_functions = [
        'random_normal', 'random_uniform', 'shuffle',
        'num_gpus', 'add', 'dev', 
        'softmax', 'softplus', 'exp',
        'tan', 'roll', 'array_equal',
        'cross_entropy', 'conv3d', 'std',
        'trace', 'full_like', 'ones_like',  
        'zeros_like', 'bitwise_invert', 'copy_array',
        'expand_dims', 'reshape', 'einsum', 'shape'
        ]

    for k, v in ivy.__dict__.items():
        if k in skip_functions:
            continue
        if k in ["namedtuple", "DType", "Dtype"] or helpers.docstring_examples_run(v):
            continue
        success = False
        failures.append(k)
    if not success:
        raise Exception(
            "the following methods had failing docstrings:\n\n{}".format(
                "\n".join(failures)
            )
        )
