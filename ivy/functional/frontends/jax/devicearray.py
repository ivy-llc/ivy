# global

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend


class DeviceArray:
    def __init__(self, array, weak_type=False):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )
        self.weak_type = weak_type

    def __repr__(self):
        main = (
            str(self.ivy_array.__repr__())
            .replace("ivy.array", "ivy.frontends.jax.DeviceArray")
            .replace(")", "")
            + ", dtype="
            + str(self.ivy_array.dtype)
        )
        if self.weak_type:
            return main + ", weak_type=True)"
        return main + ")"

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def dtype(self):
        return self.ivy_array.dtype

    @property
    def shape(self):
        return self.ivy_array.shape

    @property
    def at(self):
        return jax_frontend._src.numpy.lax_numpy._IndexUpdateHelper(self.ivy_array)

    # Instance Methods #
    # ---------------- #

    def all(self, *, axis=None, out=None, keepdims=False):
        return jax_frontend.numpy.all(
            self._ivy_array, axis=axis, keepdims=keepdims, out=out
        )

    def argmax(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):
        return jax_frontend.numpy.argmax(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def conj(self, /):
        return jax_frontend.numpy.conj(self._ivy_array)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, where=None):
        return jax_frontend.numpy.mean(
            self._ivy_array,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def cumprod(self, axis=None, dtype=None, out=None):
        return jax_frontend.numpy.cumprod(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def nonzero(self, *, size=None, fill_value=None):
        return jax_frontend.numpy.nonzero(
            self,
            size=size,
            fill_value=fill_value,
        )

    def __add__(self, other):
        return jax_frontend.numpy.add(self, other)

    def __radd__(self, other):
        return jax_frontend.numpy.add(other, self)

    def __sub__(self, other):
        return jax_frontend.lax.sub(self, other)

    def __rsub__(self, other):
        return jax_frontend.lax.sub(other, self)

    def __mul__(self, other):
        return jax_frontend.lax.mul(self, other)

    def __rmul__(self, other):
        return jax_frontend.lax.mul(other, self)

    def __div__(self, other):
        return jax_frontend.numpy.divide(self, other)

    def __rdiv__(self, other):
        return jax_frontend.numpy.divide(other, self)

    def __mod__(self, other):
        return jax_frontend.numpy.mod(self, other)

    def __rmod__(self, other):
        return jax_frontend.numpy.mod(other, self)

    def __truediv__(self, other):
        return jax_frontend.numpy.divide(self, other)

    def __rtruediv__(self, other):
        return jax_frontend.numpy.divide(other, self)

    def __matmul__(self, other):
        return jax_frontend.numpy.dot(self, other)

    def __rmatmul__(self, other):
        return jax_frontend.numpy.dot(other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return jax_frontend.lax.neg(self)

    def __eq__(self, other):
        return jax_frontend.lax.eq(self, other)

    def __ne__(self, other):
        return jax_frontend.lax.ne(self, other)

    def __lt__(self, other):
        return jax_frontend.lax.lt(self, other)

    def __le__(self, other):
        return jax_frontend.lax.le(self, other)

    def __gt__(self, other):
        return jax_frontend.lax.gt(self, other)

    def __ge__(self, other):
        return jax_frontend.lax.ge(self, other)

    def __abs__(self):
        return jax_frontend.numpy.abs(self)

    def __pow__(self, other):
        return jax_frontend.lax.pow(self, other)

    def __rpow__(self, other):
        other = ivy.asarray(other)
        return jax_frontend.lax.pow(other, self)

    def __and__(self, other):
        return jax_frontend.numpy.bitwise_and(self, other)

    def __rand__(self, other):
        return jax_frontend.numpy.bitwise_and(other, self)

    def __or__(self, other):
        return jax_frontend.numpy.bitwise_or(self, other)

    def __ror__(self, other):
        return jax_frontend.numpy.bitwise_or(other, self)

    def __xor__(self, other):
        return jax_frontend.lax.bitwise_xor(self, other)

    def __rxor__(self, other):
        return jax_frontend.lax.bitwise_xor(other, self)

    def __invert__(self):
        return jax_frontend.lax.bitwise_not(self)

    def __lshift__(self, other):
        return jax_frontend.lax.shift_left(self, other)

    def __rlshift__(self, other):
        return jax_frontend.lax.shift_left(other, self)

    def __rshift__(self, other):
        return jax_frontend.lax.shift_right_logical(self, other)

    def __rrshift__(self, other):
        return jax_frontend.lax.shift_right_logical(other, self)

    def __getitem__(self, idx):
        return self.at[idx].get()

    def __setitem__(self, idx, val):
        raise ivy.utils.exceptions.IvyException(
            "ivy.functional.frontends.jax.DeviceArray object doesn't support assignment"
        )

    def __iter__(self):
        ndim = len(self.shape)
        if ndim == 0:
            raise TypeError("iteration over a 0-d devicearray not supported")
        for i in range(ndim):
            yield self[i]
