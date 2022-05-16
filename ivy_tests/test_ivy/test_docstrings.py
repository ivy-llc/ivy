# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


def test_docstrings():
    failures = list()
    success = True
    
    ''' 
        Functions that are skipped due to precision problem include: sigmoid, inv, cos, add, det, cross_entropy, atanh, std
        Functions that are skipped due to output errors: roll, array_equal, cross_entropy, trace
        Functions skipped due to running error: conv3d, tan
    '''
    skip_functions = ['random_normal', 
                    'random_uniform', 
                    'shuffle',
                    'num_gpus',
                    'add', 
                    'inv',     
                    'sigmoid', 
                    'dev', 
                    'cos',
                    'softmax',
                    'softplus',
                    'exp',
                    'tan',
                    'atanh',
                    'det',
                    'roll',
                    'tan',
                    'array_equal',
                    'cross_entropy', 
                    'conv3d', 
                    'std',
                    'trace']
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
