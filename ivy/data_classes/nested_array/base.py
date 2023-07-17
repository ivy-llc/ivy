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
        self._pre_repr = "ivy."

    @classmethod
    def nested_array(
        cls, data, nested_rank=None, inner_shape=None, dtype=None, device=None
    ):
        dtype = ivy.default_dtype(dtype=dtype, item=data)
        device = ivy.default_device(device, item=data)

        # convert all the leaf lists to ivy arrays, determine inner_shape and depth
        def _seq_to_ivy(x, depth=0, inner_shape=None):
            inner_shape = [] if inner_shape is None else inner_shape
            if (
                isinstance(x, (list, tuple))
                and len(x) != 0
                and isinstance(x[0], (list, tuple))
            ):
                depth_ret = None
                for i, item in enumerate(x):
                    x = list(x) if isinstance(x, tuple) else x
                    ret_inner_shape = []
                    if nested_rank is not None and depth >= nested_rank:
                        ret_inner_shape = inner_shape + [len(item)]
                    x[i], depth_ret, ret_inner_shape = _seq_to_ivy(
                        item, depth=depth + 1, inner_shape=ret_inner_shape
                    )

                # We don't need to take max here,
                # because the depth will be the same for all the leafs
                depth = depth_ret
                inner_shape = ret_inner_shape
            else:
                x = ivy.array(x, dtype=dtype, device=device)
            return x, depth, inner_shape

        if isinstance(data, (list, tuple)):
            data, depth, def_inner_shape = _seq_to_ivy(data)
            depth += 1
            default_nested_rank = (
                max(0, depth - 1)
                if inner_shape is None
                else max(0, depth - 1 - len(inner_shape))
            )
            default_inner_shape = list() if nested_rank is None else def_inner_shape

            nested_rank = (
                nested_rank if nested_rank is not None else default_nested_rank
            )
            inner_shape = (
                list(inner_shape) if inner_shape is not None else default_inner_shape
            )
        elif isinstance(data, cls):
            data = data._data
            nested_rank = nested_rank if nested_rank is not None else data._nested_rank
            inner_shape = (
                list(inner_shape)
                if list(inner_shape) is not None
                else data._inner_shape
            )
        else:
            raise TypeError(
                "Input data must be pylist or tuple, got: {}".format(type(data))
            )

        return cls(data, nested_rank, inner_shape, dtype, device, internal=True)

    @staticmethod
    def nested_multi_map_in_static_method(fn_name, *args, **kwargs):
        arg_nest_idxs = ivy.nested_argwhere(
            args, ivy.is_ivy_nested_array, to_ignore=ivy.Container
        )
        kwarg_nest_idxs = ivy.nested_argwhere(
            kwargs, ivy.is_ivy_nested_array, to_ignore=ivy.Container
        )
        # retrieve all the nested_array in args and kwargs
        arg_nest = ivy.multi_index_nest(args, arg_nest_idxs)
        kwarg_nest = ivy.multi_index_nest(kwargs, kwarg_nest_idxs)
        num_nest = len(arg_nest) + len(kwarg_nest)
        fn = ivy.__dict__[fn_name]

        if num_nest == 1:
            return ivy.nested_map(
                fn,
            )

    def unbind(self):
        return tuple(ivy.copy_nest(self._data))

    def reshape(self, shape):
        assert shape[0] == self._shape[0], "batch dimension is not changeable"
        for i in range(0, shape[0]):
            new_shape = list()
            for j in range(1, len(shape)):
                if shape[j] == -1:
                    new_shape.append(self._data[i].shape[j - 1])
                else:
                    new_shape.append(shape[j])
            self._data[i] = self._data[i].reshape(new_shape)
            print(self._data[i].shape)
        self._shape = self._generate_shape()
        return self

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
        arrays_repr = "\t"
        for i in range(self._shape[0] - 1):
            arrays_repr += repr(self._data[i]) + "\n\t"
        arrays_repr += repr(self._data[-1])
        return self._pre_repr + self.__class__.__name__ + "([\n" + arrays_repr + "\n])"

    def __getitem__(self, query):
        ret = self._data[query]
        if isinstance(ret, list):
            return self.__class__.nested_array(
                ret, self._nested_rank - 1, dtype=self._dtype, device=self._device
            )
        return ret
