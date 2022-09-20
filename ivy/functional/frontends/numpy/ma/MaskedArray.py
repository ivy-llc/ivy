import ivy
import ivy.functional.frontends.numpy as np_frontend

nomask = False


class MaskedArray(np_frontend.ndarray):
    def __init__(
        self,
        data,
        mask=nomask,
        dtype=None,
        copy=False,
        ndmin=0,
        fill_value=None,
        keep_mask=True,
        hard_mask=None,
        shrink=True,
        subok=True,
        order=None,
    ):
        self._init_data_and_dtype(data, dtype)
        self._init_mask(mask)
        self._init_fill_value(fill_value)

    def _init_data_and_dtype(self, data, dtype):
        self._data = (
            ivy.array(data, dtype=dtype) if ivy.exists(dtype) else ivy.array(data)
        )
        self._dtype = self._data.dtype

    def _init_mask(self, mask):
        if isinstance(mask, list) or ivy.is_array(mask):
            ivy.assertions.check_equal(
                ivy.shape(self._data),
                ivy.shape(ivy.array(mask)),
                message="shapes of data and mask must match",
            )
            self._mask = ivy.array(mask)
        elif mask:
            self._mask = ivy.ones_like(self._data)
        else:
            self._mask = ivy.zeros_like(self._data)
        self._mask = self._mask.astype("bool")

    def _init_fill_value(self, fill_value):
        if ivy.exists(fill_value):
            self._fill_value = ivy.array(fill_value, dtype=self._dtype)
        elif ivy.is_bool_dtype(self._dtype):
            self._fill_value = ivy.array(True)
        elif ivy.is_int_dtype(self._dtype):
            self._fill_value = ivy.array(999999, dtype="int64")
        else:
            self._fill_value = ivy.array(1e20, dtype="float64")

    # Properties #
    # ---------- #

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @property
    def dtype(self):
        return self._dtype

    @property
    def fill_value(self):
        return self._fill_value

    # Setter #
    # ------ #

    @mask.setter
    def mask(self, mask):
        self._init_mask(mask)

    @dtype.setter
    def dtype(self, mask):
        # TODO: check type casting
        pass

    @fill_value.setter
    def fill_value(self, fill_value):
        self._init_fill_value(fill_value)
