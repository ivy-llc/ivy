import ivy
import ivy.functional.frontends.numpy as np_frontend
import numpy as np
from functools import reduce
from operator import mul

masked = True
nomask = False
masked_print_options = "--"

# Helpers #
# ------- #


def _is_masked_array(x):
    return isinstance(x, (np.ma.MaskedArray, np_frontend.ma.MaskedArray))


# Class #
# ----- #


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
        hard_mask=False,
        shrink=True,
        subok=True,
        order=None,
    ):
        self._init_data(data, dtype, mask, keep_mask)
        self._init_fill_value(fill_value)
        self._init_ndmin(ndmin)
        self._init_hard_mask(hard_mask)
        # shrink
        if shrink and not ivy.any(self._mask):
            self._mask = ivy.array(False)
        # copy
        if copy:
            self._data = ivy.copy_array(self._data)
            self._mask = ivy.copy_array(self._mask)
        # TODO: init super class ndarray once it's fixed

    def _init_data(self, data, dtype, mask, keep_mask):
        if _is_masked_array(data):
            self._data = (
                ivy.array(data.data, dtype=dtype)
                if ivy.exists(dtype)
                else ivy.array(data.data)
            )
            self._init_mask(mask)
            if keep_mask:
                if not isinstance(data.mask, bool):
                    ivy.assertions.check_equal(
                        ivy.shape(self._mask),
                        ivy.shape(data.mask),
                        message="shapes of input mask does not match current mask",
                    )
                self._mask = ivy.bitwise_or(self._mask, data.mask)
        else:
            self._data = (
                ivy.array(data, dtype=dtype) if ivy.exists(dtype) else ivy.array(data)
            )
            self._init_mask(mask)
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

    def _init_ndmin(self, ndmin):
        ivy.assertions.check_isinstance(ndmin, int)
        if ndmin > len(ivy.shape(self._data)):
            self._data = ivy.expand_dims(self._data, axis=0)
            self._mask = ivy.expand_dims(self._mask, axis=0)

    def _init_hard_mask(self, hard_mask):
        ivy.assertions.check_isinstance(hard_mask, bool)
        self._hard_mask = hard_mask

    # Properties #
    # ---------- #

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    # TODO: impl and check read-only?
    @property
    def recordmask(self):
        pass

    @property
    def fill_value(self):
        return self._fill_value

    # TODO (read-only)
    @property
    def sharedmask(self):
        pass

    @property
    def hardmask(self):
        return self._hard_mask

    @property
    def dtype(self):
        return self._dtype

    # Setter #
    # ------ #

    @mask.setter
    def mask(self, mask):
        self._init_mask(mask)

    @fill_value.setter
    def fill_value(self, fill_value):
        self._init_fill_value(fill_value)

    @dtype.setter
    def dtype(self, mask):
        # TODO: check type casting
        pass

    # Built-ins #
    # --------- #

    def __getitem__(self, query):
        return self._data[query]

    def __setitem__(self, query, val):
        self._data[query] = val
        if not self._hard_mask and ivy.any(self._mask):
            self._mask[query] = False

    # Instance Methods #
    # ---------------- #

    def flatten(self, order="C"):
        # TODO: return view or MA
        size = reduce(mul, ivy.shape(self._data))
        return ivy.reshape(self._data, (size,))


# masked_array (alias)
masked_array = MaskedArray
