# global

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend


class DeviceArray:
    def __init__(self, data):
        self.data = ivy.array(data) if not isinstance(data, ivy.Array) else data

    def __repr__(self):
        return (
            "ivy.functional.frontends.jax.DeviceArray("
            + str(ivy.to_list(self.data))
            + ")"
        )

    # Instance Methods #
    # ---------------- #

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
        return jax_frontend.lax.div(self, other)

    def __rdiv__(self, other):
        return jax_frontend.lax.div(other, self)

    def __mod__(self, other):
        return jax_frontend.numpy.mod(self, other)

    def __rmod__(self, other):
        return jax_frontend.numpy.mod(other, self)

    def __truediv__(self, other):
        return jax_frontend.lax.div(self, other)

    def __rtruediv__(self, other):
        return jax_frontend.lax.div(other, self)

    def __matmul__(self, other):
        return jax_frontend.numpy.dot(self, other)

    def __rmatmul__(self, other):
        return jax_frontend.numpy.dot(other, self)

    def __pos__(self):
        return ivy.positive(self)

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
