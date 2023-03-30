import ivy
import ivy.functional.frontends.paddle as paddle_frontend


class Tensor:
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    def __repr__(self):
        return (
            str(self._ivy_array.__repr__())
            .replace("ivy.array", "ivy.frontends.paddle.Tensor")
            .replace("dev", "place")
        )

    # Properties #
    # ---------- #
    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def place(self):
        return ivy.dev(self._ivy_array)

    @property
    def dtype(self):
        return self._ivy_array.dtype

    @property
    def shape(self):
        return self._ivy_array.shape

    # Setters #
    # --------#
    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

     # Instance Methods #
    # ---------------- #
    def reshape(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            return paddle_frontend.reshape(self._ivy_array, shape)
        if args:
            if isinstance(args[0], (tuple, list)):
                shape = args[0]
                return paddle_frontend.reshape(self._ivy_array, shape)
            else:
                return paddle_frontend.reshape(self._ivy_array, args)
        return paddle_frontend.reshape(self._ivy_array)


    # Implement methods

    def __getitem__(self, query):
        ret = ivy.get_item(self._ivy_array, query)
        return paddle_frontend.Tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))
