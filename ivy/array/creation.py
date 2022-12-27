# global
import abc
from numbers import Number
from typing import Optional, Union, List

# local
import ivy


# Array API Standard #
# -------------------#


class ArrayWithCreation(abc.ABC):
    def asarray(
        self: ivy.Array,
        /,
        *,
        copy: Optional[bool] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.asarray. This method simply wraps the
        function, and so the docstring for ivy.asarray also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input data, in any form that can be converted to an array. This includes
            lists, lists of tuples, tuples, tuples of tuples, tuples of lists and
            ndarrays.
        copy
            boolean, indicating whether or not to copy the input. Default: ``None``.
        dtype
            datatype, optional. Datatype is inferred from the input data.
        device
            device on which to place the created array. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array interpretation of ``self``.
        """
        return ivy.asarray(self._data, copy=copy, dtype=dtype, device=device, out=out)

    def full_like(
        self: ivy.Array,
        /,
        fill_value: float,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.full_like. This method simply wraps the
        function, and so the docstring for ivy.full_like also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        fill_value
            Scalar fill value
        dtype
            output array data type. If ``dtype`` is `None`, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and where every element is equal
            to ``fill_value``.

        Examples
        --------
        With :code:`int` datatype:

        >>> x = ivy.array([1,2,3])
        >>> fill_value = 0
        >>> x.full_like(fill_value)
        ivy.array([0, 0, 0])

        With float datatype:

        >>> fill_value = 0.000123
        >>> x = ivy.array(ivy.ones(5))
        >>> y = x.full_like(fill_value)
        >>> print(y)
        ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

        With :class:`ivy.Array` input:

        >>> x = ivy.array([1, 2, 3, 4, 5, 6])
        >>> fill_value = 1
        >>> y = x.full_like(fill_value)
        >>> print(y)
        ivy.array([1, 1, 1, 1, 1, 1])
        """
        return ivy.full_like(
            self._data, fill_value, dtype=dtype, device=device, out=out
        )

    def ones_like(
        self: ivy.Array,
        /,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.ones_like. This method simply wraps the
        function, and so the docstring for ivy.ones_like also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and filled with ones.

        """
        return ivy.ones_like(self._data, dtype=dtype, device=device, out=out)

    def zeros_like(
        self: ivy.Array,
        /,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.zeros_like. This method simply wraps
        the function, and so the docstring for ivy.zeros_like also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and filled with ``zeros``.

        """
        return ivy.zeros_like(self._data, dtype=dtype, device=device, out=out)

    def tril(
        self: ivy.Array, /, k: int = 0, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tril. This method simply wraps the
        function, and so the docstring for ivy.tril also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.
        k
            diagonal above which to zero elements. If k = 0, the diagonal is the main
            diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
            diagonal is above the main diagonal. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the lower triangular part(s). The returned array must
            have the same shape and data type as ``self``. All elements above the
            specified diagonal k must be zeroed. The returned array should be allocated
            on the same device as ``self``.

        """
        return ivy.tril(self._data, k=k, out=out)

    def triu(
        self: ivy.Array, /, k: int = 0, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.triu. This method simply wraps the
        function, and so the docstring for ivy.triu also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.    *,
        k
            diagonal below which to zero elements. If k = 0, the diagonal is the main
            diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
            diagonal is above the main diagonal. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the upper triangular part(s). The returned array must
            have the same shape and data type as ``self``. All elements below the
            specified diagonal k must be zeroed. The returned array should be allocated
            on the same device as ``self``.

        """
        return ivy.triu(self._data, k=k, out=out)

    def empty_like(
        self: ivy.Array,
        /,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.empty_like. This method simply wraps
        the function, and so the docstring for ivy.empty_like also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        dtype
            output array data type. If dtype is None, the output array data type must be
            inferred from ``self``. Deafult: ``None``.
        device
            device on which to place the created array. If device is None, the output
            array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and containing uninitialized
            data.

        """
        return ivy.empty_like(self._data, dtype=dtype, device=device, out=out)

    def meshgrid(
        self: ivy.Array,
        /,
        *arrays: Union[ivy.Array, ivy.NativeArray],
        sparse=False,
        indexing: str = "xy",
    ) -> List[ivy.Array]:
        list_arrays = [self._data] + list(arrays)
        """
        ivy.Array instance method variant of ivy.meshgrid. This method simply wraps the
        function, and so the docstring for ivy.meshgrid also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            one-dimensional input array.
        arrays
            an arbitrary number of one-dimensional arrays representing grid coordinates.
            Each array should have the same numeric data type.
        indexing
            Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero
            or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
            respectively), the ``indexing`` keyword has no effect and should be ignored.
            Default: ``'xy'``.

        Returns
        -------
        ret
            list of N arrays, where ``N`` is the number of provided one-dimensional
            input arrays. Each returned array must have rank ``N``. For ``N``
            one-dimensional arrays having lengths ``Ni = len(xi)``.
        
        """
        return ivy.meshgrid(*list_arrays, sparse, indexing=indexing)

    def from_dlpack(
        self: ivy.Array,
        /,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.from_dlpack. This method simply wraps
        the function, and so the docstring for ivy.from_dlpack also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the data in ``self``.

        """
        return ivy.from_dlpack(self._data, out=out)

    # Extra #
    # ----- #

    def copy_array(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.copy_array. This method simply wraps
        the function, and so the docstring for ivy.copy_array also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a copy of the input array ``x``.

        """
        return ivy.copy_array(self, out=out)

    def native_array(
        self: ivy.Array,
        /,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.NativeArray:
        """
        ivy.Array instance method variant of ivy.native_array. This method simply wraps
        the function, and so the docstring for ivy.native_array also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array.
        dtype
            datatype, optional. Datatype is inferred from the input data.
        device
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        ret
            A native array interpretation of ``self``.

        """
        return ivy.native_array(self._data, dtype=dtype, device=device)

    def one_hot(
        self: ivy.Array,
        depth: int,
        /,
        *,
        on_value: Optional[Number] = None,
        off_value: Optional[Number] = None,
        axis: Optional[int] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Union[ivy.Device, ivy.NativeDevice] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.one_hot. This method simply wraps the
        function, and so the docstring for ivy.one_hot also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array containing the indices for which the ones should be scattered
        depth
            Scalar defining the depth of the one-hot dimension.
        on_value
            Value to fill in output when ``indices[j] == i``. Default 1.
        off_value
            Value to fill in output when ``indices[j] != i``. Default 0.
        axis
            The axis to scatter on. The default is ``-1`` which is the last axis.
        dtype
            The data type of the output array. If None, the data type of the on_value is
            used, or if that is None, the data type of the off_value is used. Default
            float32.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            Same as x if None.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Tensor of zeros with the same shape and type as a, unless dtype provided
            which overrides.

        """
        return ivy.one_hot(
            self,
            depth,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            device=device,
            out=out,
        )

    def linspace(
        self: ivy.Array,
        stop: Union[ivy.Array, ivy.NativeArray, float],
        /,
        num: int,
        axis: int = None,
        endpoint: bool = True,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Array:
        return ivy.linspace(
            self,
            stop,
            num,
            axis=axis,
            endpoint=endpoint,
            out=out,
            dtype=dtype,
            device=device,
        )
