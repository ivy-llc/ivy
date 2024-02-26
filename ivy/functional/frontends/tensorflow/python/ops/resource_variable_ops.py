# local
import ivy.functional.frontends.tensorflow as tf_frontend


class ResourceVariable(tf_frontend.Variable):
    def __repr__(self):
        return (
            repr(self._ivy_array).replace(
                "ivy.array",
                "ivy.functional.frontends.tensorflow.python.ops.resource_variable_ops.ResourceVariable",
            )[:-1]
            + ", shape="
            + str(self._ivy_array.shape)
            + ", dtype="
            + str(self._ivy_array.dtype)
            + ")"
        )
