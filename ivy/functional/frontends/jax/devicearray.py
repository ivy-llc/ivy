# global

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


class DeviceArray:
    def __init__(self, data):
        self.data = ivy.array(data)

    def __repr__(self):
        return (
            "ivy.functional.frontends.jax.DeviceArray("
            + str(ivy.to_list(self.data))
            + ")"
        )

    # Instance Methods #
    # ---------------- #

    # Special Methods #
    # --------------- #

    @to_ivy_arrays_and_back
    def __add__(self, other):
        return jax_frontend.lax.add(self, other)

    @to_ivy_arrays_and_back
    def __radd__(self, other):
        return jax_frontend.lax.add(other, self)

    @to_ivy_arrays_and_back
    def __sub__(self, other):
        return jax_frontend.lax.sub(self, other)

    @to_ivy_arrays_and_back
    def __rsub__(self, other):
        return jax_frontend.lax.sub(other, self)

    @to_ivy_arrays_and_back
    def __mul__(self, other):
        return jax_frontend.lax.mul(self, other)

    @to_ivy_arrays_and_back
    def __rmul__(self, other):
        return jax_frontend.lax.mul(other, self)

    @to_ivy_arrays_and_back
    def __div__(self, other):
        return jax_frontend.lax.div(self, other)

    @to_ivy_arrays_and_back
    def __rdiv__(self, other):
        return jax_frontend.lax.div(other, self)

    @to_ivy_arrays_and_back
    def __mod__(self, other):
        return jax_frontend.numpy.mod(self, other)

    @to_ivy_arrays_and_back
    def __rmod__(self, other):
        return jax_frontend.numpy.mod(other, self)

    @to_ivy_arrays_and_back
    def __truediv__(self, other):
        return jax_frontend.lax.div(self, other)

    @to_ivy_arrays_and_back
    def __rtruediv__(self, other):
        return jax_frontend.lax.div(other, self)

    @to_ivy_arrays_and_back
    def __matmul__(self, other):
        return jax_frontend.lax.dot(self, other)

    @to_ivy_arrays_and_back
    def __rmatmul__(self, other):
        return jax_frontend.lax.dot(other, self)

    @to_ivy_arrays_and_back
    def __pos__(self):
        return ivy.positive(self)

    @to_ivy_arrays_and_back
    def __neg__(self):
        return jax_frontend.lax.neg(self)

    @to_ivy_arrays_and_back
    def __eq__(self, other):
        return jax_frontend.lax.eq(self, other)

    @to_ivy_arrays_and_back
    def __ne__(self, other):
        return jax_frontend.lax.ne(self, other)

    @to_ivy_arrays_and_back
    def __lt__(self, other):
        return jax_frontend.lax.lt(self, other)

    @to_ivy_arrays_and_back
    def __le__(self, other):
        return jax_frontend.lax.le(self, other)

    @to_ivy_arrays_and_back
    def __gt__(self, other):
        return jax_frontend.lax.gt(self, other)

    @to_ivy_arrays_and_back
    def __ge__(self, other):
        return jax_frontend.lax.ge(self, other)

    @to_ivy_arrays_and_back
    def __abs__(self):
        return jax_frontend.lax.abs(self)

    @to_ivy_arrays_and_back
    def __pow__(self, other):
        return jax_frontend.lax.pow(self, other)

    @to_ivy_arrays_and_back
    def __rpow__(self, other):
        return jax_frontend.lax.pow(other, self)

    @to_ivy_arrays_and_back
    def __and__(self, other):
        return jax_frontend.lax.bitwise_and(self, other)

    @to_ivy_arrays_and_back
    def __rand__(self, other):
        return jax_frontend.lax.bitwise_and(other, self)

    @to_ivy_arrays_and_back
    def __or__(self, other):
        return jax_frontend.lax.bitwise_or(self, other)

    @to_ivy_arrays_and_back
    def __ror__(self, other):
        return jax_frontend.lax.bitwise_or(other, self)

    @to_ivy_arrays_and_back
    def __xor__(self, other):
        return jax_frontend.lax.bitwise_xor(self, other)

    @to_ivy_arrays_and_back
    def __rxor__(self, other):
        return jax_frontend.lax.bitwise_xor(other, self)

    @to_ivy_arrays_and_back
    def __invert__(self):
        return jax_frontend.lax.bitwise_not(self)

    @to_ivy_arrays_and_back
    def __lshift__(self, other):
        return jax_frontend.lax.shift_left(self, other)

    @to_ivy_arrays_and_back
    def __rlshift__(self, other):
        return jax_frontend.lax.shift_left(other, self)

    @to_ivy_arrays_and_back
    def __rshift__(self, other):
        return jax_frontend.lax.shift_right_logical(self, other)

    @to_ivy_arrays_and_back
    def __rrshift__(self, other):
        return jax_frontend.lax.shift_right_logical(other, self)
