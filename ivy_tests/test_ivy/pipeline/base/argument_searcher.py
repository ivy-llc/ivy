import numpy as np


from dataclasses import dataclass

import ivy


@dataclass
class TestArgumentsSearchResult:
    original: np.array
    values: np.array
    indices: list


class ArgumentsSearcher:
    args: list
    kwargs: dict

    def __init__(self, test_arguments):
        self.test_arguments = test_arguments

    def _split_args_to_args_and_kwargs(self, num_positional_args):
        """
        Split the kwargs into args and kwargs.

        The first num_positional_args ported to args.
        """
        args = [v for v in list(self.test_arguments.values())[:num_positional_args]]
        kwargs = {
            k: self.test_arguments[k]
            for k in list(self.test_arguments.keys())[num_positional_args:]
        }
        return args, kwargs

    def _get_nested_np_arrays(self, nest):
        """
        Search for a NumPy arrays in a nest.

        Parameters
        ----------
        nest
            nest to search in.

        Returns
        -------
            Items found, indices, and total number of arrays found
        """
        indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))

        ret = ivy.multi_index_nest(nest, indices)
        return ret, indices, len(ret)

    def search_args(self, num_positional_args):
        # split the arguments into their positional and keyword components
        args_np, kwargs_np = self._split_args_to_args_and_kwargs(
            num_positional_args=num_positional_args,
            # test_arguments=self.test_arguments,
        )

        # Extract all arrays from the arguments and keyword arguments
        arg_np_arrays, arrays_args_indices, n_args_arrays = self._get_nested_np_arrays(
            args_np
        )
        kwarg_np_arrays, arrays_kwargs_indices, n_kwargs_arrays = (
            self._get_nested_np_arrays(kwargs_np)
        )

        total_num_arrays = n_args_arrays + n_kwargs_arrays
        args_result = TestArgumentsSearchResult(
            args_np, arg_np_arrays, arrays_args_indices
        )
        kwargs_result = TestArgumentsSearchResult(
            kwargs_np, kwarg_np_arrays, arrays_kwargs_indices
        )
        return args_result, kwargs_result, total_num_arrays
