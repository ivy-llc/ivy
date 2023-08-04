# global
import abc
from typing import List, Tuple

# local
import ivy


class NestedArrayBase(abc.ABC):
    """Base class for nested array objects."""

    def __init__(self, data, nested_rank, inner_shape, dtype, device, internal=False):
        if not internal:
            raise RuntimeError(
                "NestedArray is an abstract class "
                "and should not be instantiated directly."
                "Please use one of the factory methods instead"
            )
        self._data = data
        self._nested_rank = nested_rank
        self._inner_shape = inner_shape
        self._shape = [len(self._data)] + [None] * self._nested_rank + self._inner_shape
        self._dtype = dtype
        self._device = device
        self._pre_repr = "ivy.NestedArray"

    @classmethod
    def nested_array(
        cls, data, nested_rank=None, inner_shape=None, dtype=None, device=None
    ):
        dtype = ivy.default_dtype(dtype=dtype, item=data)
        device = ivy.default_device(device, item=data)

        # convert all the leaf lists to ivy arrays, determine inner_shape and depth
        det_inner_shape = list()

        # ToDo: add check for depth being the same for all nests
        def _seq_to_ivy(x, depth=0):
            if nested_rank is not None and depth >= nested_rank:
                x = ivy.array(x, dtype=dtype, device=device)
                depth += x.ndim - 1
                if x.ndim > 1:
                    det_inner_shape.append(list(x.shape[1:]))
                else:
                    det_inner_shape.append(list())
            elif (
                isinstance(x, (list, tuple))
                and len(x) != 0
                and isinstance(x[0], (list, tuple))
            ):
                depth_ret = None
                for i, item in enumerate(x):
                    x = list(x) if isinstance(x, tuple) else x
                    x[i], depth_ret = _seq_to_ivy(item, depth=depth + 1)

                depth = depth_ret if depth_ret else depth
            else:
                x = ivy.array(x, dtype=dtype, device=device)
                if x.ndim > 1:
                    det_inner_shape.append(list(x.shape[1:]))
                else:
                    det_inner_shape.append(list())
            return x, depth

        if isinstance(data, (list, tuple)):
            data, depth = _seq_to_ivy(data)
            depth += 1
            # make sure that all the elements of det_inner_shape are the same
            if len(det_inner_shape) > 0:
                if [det_inner_shape[0]] * len(det_inner_shape) != det_inner_shape:
                    raise ValueError(
                        "All the elements of the nested array must have the same "
                        "inner shape, got: {}".format(det_inner_shape)
                    )
                det_inner_shape = det_inner_shape[0]

            # defining default values for nested_rank and inner_shape
            default_nested_rank = (
                max(0, depth - 1)
                if inner_shape is None
                else max(0, depth - 1 - len(inner_shape))
            )
            default_inner_shape = list() if nested_rank is None else det_inner_shape

            # determining actual values for nested_rank and inner_shape
            nested_rank = (
                nested_rank if nested_rank is not None else default_nested_rank
            )
            inner_shape = (
                list(inner_shape) if inner_shape is not None else default_inner_shape
            )
        elif isinstance(data, cls):
            data = data._data
            nested_rank = nested_rank if nested_rank is not None else data.nested_rank
            inner_shape = (
                list(inner_shape) if list(inner_shape) is not None else data.inner_shape
            )
        else:
            raise TypeError(
                "Input data must be pylist or tuple, got: {}".format(type(data))
            )

        return cls(data, nested_rank, inner_shape, dtype, device, internal=True)

    @staticmethod
    def ragged_multi_map_in_function(fn, *args, **kwargs):
        arg_nest_idxs = ivy.nested_argwhere(
            args, ivy.is_ivy_nested_array, to_ignore=ivy.NestedArray
        )
        kwarg_nest_idxs = ivy.nested_argwhere(
            kwargs, ivy.is_ivy_nested_array, to_ignore=ivy.NestedArray
        )
        # retrieve all the nested_array in args and kwargs
        arg_nest = ivy.multi_index_nest(args, arg_nest_idxs)
        kwarg_nest = ivy.multi_index_nest(kwargs, kwarg_nest_idxs)
        num_arg_nest, num_kwarg_nest = len(arg_nest), len(kwarg_nest)
        num_nest = num_arg_nest + num_kwarg_nest
        inspect_fn = fn
        if isinstance(fn, str):
            inspect_fn = ivy.__dict__[fn]
        nests = arg_nest + kwarg_nest

        def map_fn(vals):
            arg_vals = vals[:num_arg_nest]
            a = ivy.copy_nest(args, to_mutable=True)
            ivy.set_nest_at_indices(a, arg_nest_idxs, arg_vals)
            kwarg_vals = vals[num_arg_nest:]
            kw = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.set_nest_at_indices(kw, kwarg_nest_idxs, kwarg_vals)
            return inspect_fn(*a, **kw)

        if num_nest == 0:
            raise Exception(
                "No RaggedArrays found in args or kwargs of function {}".format(fn)
            )
        ret = ivy.NestedArray.ragged_multi_map(map_fn, nests)
        return ret

    @staticmethod
    def ragged_multi_map(fn, ragged_arrays):
        args = list()
        for ragged in ragged_arrays:
            args.append(ivy.copy_nest(ragged.data))
        ragged_arrays[0]
        ret = ivy.nested_multi_map(lambda x, _: fn(x), args)
        # infer dtype, shape, and device from the first array in the ret data
        broadcasted_shape = ivy.NestedArray.broadcast_shapes(
            [arg.shape for arg in ragged_arrays]
        )
        # infer ragged_rank from broadcasted shape
        for i, dim in enumerate(broadcasted_shape[::-1]):
            if dim is None:
                nested_rank = len(broadcasted_shape) - i - 1
                break
        inner_shape = broadcasted_shape[nested_rank:]
        arr0_id = ivy.nested_argwhere(ret, ivy.is_ivy_array, stop_after_n_found=1)[0]
        arr0 = ivy.index_nest(ret, arr0_id)
        ragged_ret = ivy.NestedArray.nested_array(
            ret, nested_rank, inner_shape, arr0.dtype, arr0.device
        )
        return ragged_ret

    @staticmethod
    def replace_ivy_arrays(ragged_array, arrays):
        data = ragged_array.data
        ivy_idxs = ivy.nested_argwhere(data, ivy.is_ivy_array)
        arr0 = arrays[0]
        inner_shape, dev, dtype = arr0.shape.as_list(), arr0.device, arr0.dtype
        ret = ivy.set_nest_at_indices(data, ivy_idxs, arrays, shallow=False)
        return ivy.NestedArray.nested_array(
            ret, ragged_array.nested_rank, inner_shape, dtype, dev
        )

    @staticmethod
    def broadcast_shapes(shapes):
        z = []
        max_length = max([len(x) for x in shapes])
        shape_list = list(shapes)
        # making every shape the same length
        for i, shape in enumerate(shapes):
            if len(shape) != max_length:
                shape_list[i] = [1] * (max_length - len(shape)) + shape
        # broadcasting
        for x in zip(*shape_list):
            if None in x:
                for dims in x:
                    if dims is not None and dims != 1:
                        raise ValueError(
                            "Shapes {} and {} are not broadcastable".format(
                                shapes[0], shapes[1]
                            )
                        )
                z.append(None)
            elif 1 in x:
                dim_exist = False
                for dims in x:
                    if dims != 1:
                        z.append(dims)
                        if dim_exist:
                            raise ValueError(
                                "Shapes {} and {} are not broadcastable".format(
                                    shapes[0], shapes[1]
                                )
                            )
                        dim_exist = True
                if not dim_exist:
                    z.append(1)
            else:
                if len(set(x)) == 1:
                    z.append(x[0])
                else:
                    raise ValueError(
                        "Shapes {} and {} are not broadcastable".format(
                            shapes[0], shapes[1]
                        )
                    )
        return z

    def ragged_map(self, fn):
        arg = ivy.copy_nest(self._data)
        ivy.nested_map(arg, lambda x: fn(x), shallow=True)
        # infer dtype, shape, and device from the first array in the ret data
        arr0_id = ivy.nested_argwhere(arg, ivy.is_ivy_array, stop_after_n_found=1)[0]
        arr0 = ivy.index_nest(arg, arr0_id)
        inner_shape = arr0.shape.as_list()[1:]
        ragged_ret = ivy.NestedArray.nested_array(
            arg, self._nested_rank, inner_shape, arr0.dtype, arr0.device
        )
        return ragged_ret

    def unbind(self):
        return tuple(ivy.copy_nest(self._data))

    # Properties #
    # ---------- #

    @property
    def data(self) -> ivy.NativeArray:
        """The native array being wrapped in self."""
        return self._data

    @property
    def dtype(self) -> ivy.Dtype:
        """Data type of the array elements."""
        return self._dtype

    @property
    def device(self) -> ivy.Device:
        """Hardware device the array data resides on."""
        return self._device

    @property
    def shape(self) -> List:
        """Array dimensions."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes)."""
        return len(tuple(self._shape))

    @property
    def nested_rank(self) -> int:
        """Nested Rank."""
        return self._nested_rank

    @property
    def inner_shape(self) -> Tuple[int]:
        """Inner Shape."""
        return self._inner_shape

    # Built-ins #
    # ----------#

    def __repr__(self):
        rep = self._data.__repr__().replace("[ivy.array", "[")
        rep = rep.replace("ivy.array", "\n\t").replace("(", "").replace(")", "")
        ret = self._pre_repr + "(\n\t" + rep + "\n)"
        return ret

    def __getitem__(self, query):
        ret = self._data[query]
        if isinstance(ret, list):
            return self.__class__.nested_array(
                ret, self._nested_rank - 1, dtype=self._dtype, device=self._device
            )
        return ret
