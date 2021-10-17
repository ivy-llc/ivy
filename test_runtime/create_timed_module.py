import os
import json
import argparse

this_file_dir = os.path.dirname(os.path.realpath(__file__))
ivy_dir = os.path.join(this_file_dir, '../ivy')
with_time_logs_dir = os.path.join(this_file_dir, '../with_time_logs')

OFFSETS = {'general':
               {'jax': {'unstack': [-2], 'indices_where': [-2], 'one_hot': [None], 'identity': [-7],
                        'scatter_flat': [-4, -5, -8, -9, -11, -12], 'scatter_nd': [-4, -5, -8, -9, -11, -12],
                        'gather_nd': [-3], 'get_device': [-1]},
                'mxnet': {'linspace': [-7, -9, -11], 'zero_pad': [-3], 'indices_where': [-5], 'zeros_like': [-1],
                         'ones_like': [-1], 'cross': [None], 'matmul': [-1], 'identity': [-7]},
                'numpy': {'unstack': [-2], 'indices_where': [-2], 'one_hot': [None], 'identity': [-7],
                          'scatter_flat': [-4, -5, -8, -9, -11, -12], 'scatter_nd': [-4, -5, -8, -9, -11, -12],
                          'gather_nd': [-3]},
                'tensorflow': {'scatter_flat': [-3], 'scatter_nd': [-4], 'get_device': [-1]},
                'torch': {'linspace': [-6, -8, -10], 'indices_where': [-2], 'identity': [-7],
                          'scatter_flat': [-2, -9, -12, -14, -16], 'scatter_nd': [-4, -18, -21, -23, -25],
                          'gather_nd': [-2], 'get_device': [-1]}},
           'gradients': {'jax': {'execute_with_gradients': [-1, -10]},
                         'mxnet': {'variable': [-1], 'execute_with_gradients': [-1, -8],
                                  'gradient_descent_update': [None]},
                         'numpy': {'variable': [None], 'execute_with_gradients': [None], 'stop_gradient': [None]},
                         'tensorflow': {'execute_with_gradients': [-1, -8], 'gradient_descent_update': [None]},
                         'torch': {'variable': [-1], 'execute_with_gradients': [-1, -8],
                                   'gradient_descent_update': [None],
                                   'stop_gradient': [-1]}},
           'image': {'jax': {'stack_images': [None], 'bilinear_resample': [None], 'gradient_image': [None]},
                     'mxnet': {'stack_images': [None], 'bilinear_resample': [-2], 'gradient_image': [None]},
                     'numpy': {'stack_images': [None], 'bilinear_resample': [None], 'gradient_image': [None]},
                     'tensorflow': {'stack_images': [None], 'bilinear_resample': [-1], 'gradient_image': [None]},
                     'torch': {'stack_images': [None], 'bilinear_resample': [-2], 'gradient_image': [None]}},
           'linalg': {'jax': {'vector_to_skew_symmetric_matrix': [None]},
                      'mxnet': {'svd': [-1], 'pinv': [None], 'vector_to_skew_symmetric_matrix': [None]},
                      'numpy': {'vector_to_skew_symmetric_matrix': [None]},
                      'tensorflow': {'svd': [-2], 'vector_to_skew_symmetric_matrix': [None]},
                      'torch': {'svd': [-2], 'vector_to_skew_symmetric_matrix': [None]}},
           'logic': {'jax': {},
                     'mxnet': {},
                     'numpy': {},
                     'tensorflow': {},
                     'torch': {}},
           'math': {'jax': {},
                     'mxnet': {},
                     'numpy': {},
                     'tensorflow': {},
                     'torch': {}},
           'random': {'jax': {'random_uniform': [None], 'randint': [None], 'seed': [None], 'shuffle': [None]},
                      'mxnet': {},
                      'numpy': {'shuffle': [None]},
                      'tensorflow': {},
                      'torch': {'seed': [-1, -2]}},
           'reductions': {'jax': {},
                          'mxnet': {},
                          'numpy': {},
                          'tensorflow': {},
                          'torch': {'reduce_prod': [-1], 'reduce_min': [-1], 'reduce_max': [-1]}},
           'activations': {'jax': {'softmax': [None]},
                           'mxnet': {},
                           'numpy': {'softmax': [None]},
                           'tensorflow': {},
                           'torch': {}},
           'layers': {'jax': {},
                      'mxnet': {'conv1d': [-4], 'conv1d_transpose': [-4], 'conv2d': [-4], 'conv2d_transpose': [-4],
                               'depthwise_conv2d': [-4], 'conv3d': [-4], 'conv3d_transpose': [-4]},
                      'numpy': {},
                      'tensorflow': {},
                      'torch': {'conv1d': [-1], 'conv1d_transpose': [-1], 'conv2d': [-1], 'conv2d_transpose': [-1],
                                'depthwise_conv2d': [-1], 'conv3d': [-1], 'conv3d_transpose': [-1]}}}

NAMESPACE_DICT = {'jax': ['_jnp.', '_jax.'],
                  'mxnet': ['_mx.'],
                  'numpy': ['_np.'],
                  'tensorflow': ['_tf.'],
                  'torch': ['_torch.']}


def main(dim):
    os.makedirs(with_time_logs_dir, exist_ok=True)
    os.system('rsync -rav {} {}'.format(ivy_dir, with_time_logs_dir))
    from test_runtime.utils import modify_ivy_file
    submodules_dict = {'core': ['general', 'gradients', 'image', 'linalg', 'logic', 'math', 'random', 'reductions'],
                       'nn': ['activations', 'layers']}
    for framework in ['jax', 'mxnet', 'numpy', 'tensorflow', 'torch']:
        for subdir, submodules in submodules_dict.items():
            for submodule in submodules:
                modify_ivy_file(os.path.join(with_time_logs_dir, 'ivy/{}/{}/{}.py'.format(
                    framework, subdir, submodule)), os.path.join(this_file_dir, 'test_{}_runtime'.format(subdir)),
                                OFFSETS[submodule], NAMESPACE_DICT, dim)
                with open(os.path.join(this_file_dir, 'test_{}_runtime/runtime_analysis/{}/{}_offsets.json'.format(
                        subdir, dim, submodule)), 'w+') as file:
                    file.write(json.dumps(OFFSETS[submodule], indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dim', type=int, default=int(1e4))
    main(parser.parse_args().dim)
