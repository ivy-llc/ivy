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

        def _seq_to_ivy(x, depth=0):
            if nested_rank is not None and depth >= nested_rank:
                x = ivy.array(x, dtype=dtype, device=device)
                depth += x.ndim - 1
                det_inner_shape.append(list(x.shape[1:]))
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
                det_inner_shape.append(list(x.shape[1:]))
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
    def nested_multi_map_in_function(fn_name, *args, **kwargs):
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
