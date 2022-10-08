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
        return jax_frontend.add(self.data, other)

    @to_ivy_arrays_and_back
    def __radd__(self, other):
        return jax_frontend.add(other, self.data)

    @to_ivy_arrays_and_back
    def __sub__(self, other):
        return jax_frontend.sub(self.data, other)

    @to_ivy_arrays_and_back
    def __rsub__(self, other):
        return jax_frontend.sub(other, self.data)

    @to_ivy_arrays_and_back
    def __mul__(self, other):
        return jax_frontend.mul(self.data, other)

    @to_ivy_arrays_and_back
    def __rmul__(self, other):
        return jax_frontend.mul(other, self.data)

    @to_ivy_arrays_and_back
    def __div__(self, other):
        return jax_frontend.div(self.data, other)

    @to_ivy_arrays_and_back
    def __rdiv__(self, other):
        return jax_frontend.div(other, self.data)

    @to_ivy_arrays_and_back
    def __mod__(self, other):
        return jax_frontend.mod(self.data, other)

    @to_ivy_arrays_and_back
    def __rmod__(self, other):
        return jax_frontend.mod(other, self.data)

    @to_ivy_arrays_and_back
    def __divmod__(self, other):
        return jax_frontend.divmod(self.data, other)

    @to_ivy_arrays_and_back
    def __rdivmod__(self, other):
        return jax_frontend.divmod(other, self.data)

    @to_ivy_arrays_and_back
    def __truediv__(self, other):
        return jax_frontend.div(self.data, other)

    @to_ivy_arrays_and_back
    def __rtruediv__(self, other):
        return jax_frontend.div(other, self.data)

    @to_ivy_arrays_and_back
    def __floordiv__(self, other):
        return jax_frontend.floor_divide(self.data, other)

    @to_ivy_arrays_and_back
    def __rfloordiv__(self, other):
        return jax_frontend.floor_divide(other, self.data)

    @to_ivy_arrays_and_back
    def __matmul__(self, other):
        return jax_frontend.dot(self.data, other)

    @to_ivy_arrays_and_back
    def __rmatmul__(self, other):
        return jax_frontend.dot(other, self.data)

    @to_ivy_arrays_and_back
    def __pos__(self):
        return ivy.positive(self)

    @to_ivy_arrays_and_back
    def __neg__(self):
        return jax_frontend.neg(self)

    @to_ivy_arrays_and_back
    def __eq__(self, other):
        return jax_frontend.eq(self, other)

    @to_ivy_arrays_and_back
    def __ne__(self, other):
        return jax_frontend.ne(self, other)

    @to_ivy_arrays_and_back
    def __lt__(self, other):
        return jax_frontend.lt(self, other)

    @to_ivy_arrays_and_back
    def __le__(self, other):
        return jax_frontend.le(self, other)

    @to_ivy_arrays_and_back
    def __gt__(self, other):
        return jax_frontend.gt(self, other)

    @to_ivy_arrays_and_back
    def __ge__(self, other):
        return jax_frontend.ge(self, other)

    @to_ivy_arrays_and_back
    def __abs__(self):
        return jax_frontend.abs(self)

    @to_ivy_arrays_and_back
    def __pow__(self, other):
        return jax_frontend.pow(self, other)

    @to_ivy_arrays_and_back
    def __rpow__(self, other):
        return jax_frontend.pow(other, self)

    @to_ivy_arrays_and_back
    def __and__(self, other):
        return jax_frontend.bitwise_and(self, other)

    @to_ivy_arrays_and_back
    def __rand__(self, other):
        return jax_frontend.bitwise_and(other, self)

    @to_ivy_arrays_and_back
    def __or__(self, other):
        return jax_frontend.bitwise_or(self, other)

    @to_ivy_arrays_and_back
    def __ror__(self, other):
        return jax_frontend.bitwise_or(other, self)

    @to_ivy_arrays_and_back
    def __xor__(self, other):
        return jax_frontend.bitwise_xor(self, other)

    @to_ivy_arrays_and_back
    def __rxor__(self, other):
        return jax_frontend.bitwise_xor(other, self)

    @to_ivy_arrays_and_back
    def __invert__(self):
        return jax_frontend.bitwise_not(self)

    @to_ivy_arrays_and_back
    def __lshift__(self, other):
        return jax_frontend.shift_left(self, other)

    @to_ivy_arrays_and_back
    def __rlshift__(self, other):
        return jax_frontend.shift_left(other, self)

    @to_ivy_arrays_and_back
    def __rshift__(self, other):
        return jax_frontend.shift_right_logical(self, other)

    @to_ivy_arrays_and_back
    def __rrshift__(self, other):
        return jax_frontend.shift_right_logical(other, self)
