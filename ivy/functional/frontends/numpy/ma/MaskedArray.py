import ivy
import ivy.functional.frontends.numpy as np_frontend
import numpy as np


masked = True
masked_print_options = "--"
nomask = False


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
                    ivy.utils.assertions.check_equal(
                        ivy.shape(self._mask),
                        ivy.shape(data.mask),
                        message="shapes of input mask does not match current mask",
                        as_array=False,
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
            ivy.utils.assertions.check_equal(
                ivy.shape(self._data),
                ivy.shape(ivy.array(mask)),
                message="shapes of data and mask must match",
                as_array=False,
            )
            self._mask = ivy.array(mask)
        elif mask.all():
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
        ivy.utils.assertions.check_isinstance(ndmin, int)
        if ndmin > len(ivy.shape(self._data)):
            self._data = ivy.expand_dims(self._data, axis=0)
            self._mask = ivy.expand_dims(self._mask, axis=0)

    def _init_hard_mask(self, hard_mask):
        ivy.utils.assertions.check_isinstance(hard_mask, bool)
        self._hard_mask = hard_mask

    # Properties #
    # ---------- #

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @property
    def fill_value(self):
        return self._fill_value

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

    # Built-ins #
    # --------- #

    def __getitem__(self, query):
        if self._mask.shape != self._data.shape:
            self._mask = ivy.ones_like(self._data, dtype=ivy.bool) * self._mask
        if self._fill_value.shape != self._data.shape:
            self._fill_value = ivy.ones_like(self._data) * self._fill_value
        if hasattr(self._mask[query], "shape"):
            return MaskedArray(
                data=self._data[query],
                mask=self._mask[query],
                fill_value=self._fill_value[query],
                hard_mask=self._hard_mask,
            )

    def __setitem__(self, query, val):
        self._data[query] = val
        if self._mask.shape != self._data.shape:
            self._mask = ivy.ones_like(self._data, dtype=ivy.bool) * self._mask
        val_mask = ivy.ones_like(self._mask[query]) * getattr(val, "_mask", False)
        if self._hard_mask:
            self._mask[query] |= val_mask
        else:
            self._mask[query] = val_mask
        return self

    def __repr__(self):
        dec_vals = ivy.array_decimal_values
        with np.printoptions(precision=dec_vals):
            return (
                "ivy.MaskedArray("
                + self._array_in_str()
                + ",\n\tmask="
                + str(self._mask.to_list())
                + ",\n\tfill_value="
                + str(self._fill_value.to_list())
                + "\n)"
            )

    def _array_in_str(self):
        # check if we have unsized array
        if self._data.shape == ():
            if self._mask:
                return masked_print_options
            return str(self._data.to_list())
        if ivy.any(self._mask):
            return str(
                [
                    masked_print_options if mask else x
                    for x, mask in zip(self._data.to_list(), self._mask.to_list())
                ]
            )
        return str(self._data.to_list())


# --- Helpers --- #
# --------------- #


def _is_masked_array(x):
    return isinstance(x, (np.ma.MaskedArray, np_frontend.ma.MaskedArray))


# Instance Methods #
# ---------------- #

# TODO


# masked_array (alias)
masked_array = MaskedArray
