# global
import weakref

# local
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow import EagerTensor


class TensorArray:
    def __init__(
        self,
        dtype,
        size=None,
        dynamic_size=None,
        clear_after_read=None,
        tensor_array_name=None,
        handle=None,
        flow=None,
        infer_shape=True,
        element_shape=None,
        colocate_with_first_write_call=True,
        name=None,
    ):
        del (flow, tensor_array_name, name)
        self._handle = None
        self._flow = tf_frontend.constant(0, dtype=tf_frontend.int32)
        self._infer_shape = infer_shape
        self._element_shape = (
            ivy.Shape(element_shape) if element_shape is not None else element_shape
        )
        self._colocate_with_first_write_call = colocate_with_first_write_call
        self._dtype = tf_frontend.as_dtype(dtype)
        self._dynamic_size = dynamic_size or False
        self._clear_after_read = True if clear_after_read is None else clear_after_read
        self._previously_read_indices = []

        if isinstance(size, EagerTensor):
            size = size.ivy_array
        self._tensor_array = [None for _ in range(size)]
        self._parent = weakref.ref(self)

    @property
    def flow(self):
        return self._flow

    @property
    def dtype(self):
        return self._dtype

    @property
    def handle(self):
        return self._handle

    @property
    def element_shape(self):
        return self._element_shape

    def identity(self):
        return self._parent()

    def grad(self, source, flow=None, name=None):
        raise NotImplementedError(
            "TensorArray.grad is not supported when executing eagerly; eager's "
            "gradient implementation does not use/need this function to compute "
            "gradients of operations that use TensorArrays."
        )

    @property
    def dynamic_size(self):
        return self._dynamic_size

    @property
    def infer_shape(self):
        return self._infer_shape

    def read(self, index, name=None):
        if isinstance(index, EagerTensor):
            index = ivy.to_scalar(index.ivy_array)

        if index < 0:
            raise IndexError(f"Reading from negative indices {index} is not allowed.")

        if index >= len(self._tensor_array):
            raise IndexError(
                f"Tried to read from index {index} but array size is:"
                f" {len(self._tensor_array)} "
            )

        tensor = self._tensor_array[index]
        if tensor is None:
            if index in self._previously_read_indices:
                raise ValueError(
                    f"Could not read index {index} twice because it was cleared after a"
                    " previous read (perhaps try setting clear_after_read = false?)"
                )
            else:
                tensor = self._tensor_array[index] = tf_frontend.zeros(
                    shape=self._element_shape, dtype=self._dtype
                )

        if self._clear_after_read:
            self._tensor_array[index] = None
            self._previously_read_indices.append(index)
        return tensor

    def _write(self, index, value, name=None):
        if isinstance(index, EagerTensor):
            index = ivy.to_scalar(index.ivy_array)

        if index < 0:
            raise IndexError(f"Reading from negative indices {index} is not allowed.")

        size = len(self._tensor_array)
        if index >= size:
            if not self._dynamic_size:
                raise IndexError(
                    "Tried to write to index {index} but array is not resizeable and"
                    " size is: {size}"
                )
            self._tensor_array.extend(None for _ in range(index - size + 1))

        if not isinstance(value, EagerTensor):
            value = tf_frontend.cast(value, self.dtype)

        if self._dtype != value.dtype:
            raise ValueError(
                f"TensorArray dtype is {self._dtype} but Op is trying to write dtype"
                f" {value.dtype} "
            )

        if self._infer_shape:
            self._element_shape = self._merge_shape(value)

        self._tensor_array[index] = value

    def _merge_shape(self, value):
        if self._element_shape is None:
            return value.shape
        if len(self._element_shape) != len(value.shape):
            raise ValueError("Shapes not compatible")
        shape = []
        for a, b in zip(self._element_shape, value.shape):
            if a == b or a is None:
                shape.append(b)
            else:
                raise ValueError("Shapes not compatible")
        return tuple(shape)

    def write(self, index, value, name=None):
        self._write(index, value)
        return self._parent()

    def stack(self, name=None):
        if self._tensor_array:
            for ix in range(len(self._tensor_array)):
                if self._tensor_array[ix] is None:
                    self._tensor_array[ix] = tf_frontend.zeros(
                        shape=self._element_shape, dtype=self._dtype
                    )
        if not self._tensor_array and self._element_shape.is_fully_defined():
            return tf_frontend.constant(
                [0] + list(self.element_shape), dtype=self._dtype
            )
        else:
            return tf_frontend.stack(self._tensor_array)

    def _maybe_zero(self, ix):
        val = self._tensor_array[ix]
        if val is None:
            val = self._tensor_array[ix] = tf_frontend.zeros(
                shape=self._element_shape, dtype=self._dtype
            )
        return val

    def gather(self, indices, name=None):
        if isinstance(indices, EagerTensor):
            indices = indices.ivy_array
        return tf_frontend.stack([self._maybe_zero(i) for i in indices])

    def concat(self, name=None):
        return tf_frontend.concat(
            [self._maybe_zero(ix) for ix in range(len(self._tensor_array))],
            0,
            name=name,
        )

    def unstack(self, value, name=None):
        tensors = tf_frontend.unstack(value, name=name)
        if len(tensors) > len(self._tensor_array) and not self._dynamic_size:
            raise ValueError(
                f"Cannot unstack {len(tensors)} tensors into a TensorArray of static"
                f" size {len(self._tensor_array)} "
            )
        self._tensor_array = tensors
        return self._parent()

    def scatter(self, indices, value, name=None):
        if isinstance(indices, EagerTensor):
            indices = indices.ivy_array
        for index, val in zip(indices, tf_frontend.unstack(value)):
            self._write(index, val)
        return self._parent()

    def size(self, name=None):
        return tf_frontend.constant(len(self._tensor_array))

    def close(self, name=None):
        del self._tensor_array[:]

    def split(self, value, lengths, name=None):
        value = tf_frontend.cast(value, self.dtype)
        lengths = (
            tf_frontend.constant(lengths)
            if not isinstance(lengths, EagerTensor)
            else lengths
        )
        self._tensor_array = tf_frontend.split(value, lengths, name=name)
        return self._parent()
