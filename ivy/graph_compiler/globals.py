import ivy
import logging
wrapping_paused = False
op_logging = False
wrapped_stack = list()
raw_pids_to_weakrefs = dict()
raw_pids_to_unique_pids = dict()
dependent_pids = set()
time_inference = False
timing_fname = None
sum_inference_times = {'0_init_param_setting': 0,
                       '1_pre_param_setting': 0,
                       '2_fn_call': 0,
                       '2_0_arg_n_kwarg_copying': 0,
                       '2_1_arg_n_kwarg_writing': 0,
                       '2_2_backend_fn': 0,
                       '3_post_param_setting': 0,
                       '4_end_param_setting': 0,
                       'total': 0,
                       'count': 0}


def log_global_inference_abs_times():
    if timing_fname is None:
        logging.info('abs times: {}'.format(
            ivy.Container({k: v / sum_inference_times['count'] for k, v in sum_inference_times.items()})))
    # noinspection PyTypeChecker
    with open(timing_fname, 'w+') as f:
        f.write('abs times: {}\n'.format(
            str(ivy.Container({k: v/sum_inference_times['count'] for k, v in sum_inference_times.items()}))))


def log_global_inference_rel_times():
    if timing_fname is None:
        logging.info('relative times: {}'.format(
            ivy.Container({k: v / sum_inference_times['total'] for k, v in sum_inference_times.items()})))
    # noinspection PyTypeChecker
    with open(timing_fname, 'w+') as f:
        f.write('relative times: {}\n'.format(
            str(ivy.Container({k: v / sum_inference_times['total'] for k, v in sum_inference_times.items()}))))


ARRAY_BUILTINS = ['__neg__', '__pow__', '__rpow__', '__add__', '__radd__', '__iadd__', '__sub__', '__rsub__',
                  '__isub__', '__mul__', '__rmul__', '__imul__', '__truediv__', '__rtruediv__', '__itruediv__',
                  '__floordiv__', '__rfloordiv__', '__ifloordiv__', '__abs__', '__lt__', '__le__', '__eq__', '__ne__',
                  '__gt__', '__ge__', '__and__', '__rand__', '__or__', '__ror__', '__invert__', '__xor__', '__rxor__',
                  '__getitem__', '__setitem__', '__getattr__', '__setattr__', '__getattribute__']

CLASSES_TO_WRAP = {'numpy': [],
                   'jax': [],
                   'tensorflow': [],
                   'torch': [('torch', 'Tensor')],
                   'mxnet': []}

GRAPH_ATTRIBUTES = {'numpy': [],
                   'jax': [],
                   'tensorflow': [],
                   'torch': ['data', 'requires_grad'],
                   'mxnet': []}

GENERATOR_METHODS = {'numpy': [],
                     'jax': [],
                     'tensorflow': [],
                     'torch': ['rand'],
                     'mxnet': []}
