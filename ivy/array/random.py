# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithRandom(abc.ABC):
    def random_uniform(
        self: ivy.Array,
        high: Union[float, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.random_uniform. This method simply
        wraps the function, and so the docstring for ivy.random_uniform also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Lower boundary of the output interval. All values generated will be
            greater than or equal to ``low``. If array, must have same shape as
            ``high``.
        high
            Upper boundary of the output interval. All the values generated will be
            less than ``high``. If array, must have same shape as ``low``.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``low`` and ``high`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default floating-point data type. Default ``None``
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized uniform distribution.

        Examples
        --------
        >>> x = ivy.array([[9.8, 3.4], [5.8, 7.2]])
        >>> x.random_uniform(10.2)
        ivy.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x = ivy.array([8.7, 9.3])
        >>> y = ivy.array([12.8, 14.5])
        >>> z = ivy.ones((2,))
        >>> x.random_uniform(y, device='cpu', dtype='float64', out=z)
        ivy.array([12.1, 14. ])
        """
        return ivy.random_uniform(
            self._data,
            high,
            shape,
            device=device,
            dtype=dtype,
            out=out,
        )

    def random_normal(
        self: ivy.Array,
        std: Union[float, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.random_uniform. This method simply
        wraps the function, and so the docstring for ivy.random_uniform also applies to
        this method with minimal changes.

        Parameters
        ----------
        self (to update)
            Lower boundary of the output interval. All values generated will be
            greater than or equal to ``low``. If array, must have same shape as
            ``high``.
        high (to update)
            Upper boundary of the output interval. All the values generated will be
            less than ``high``. If array, must have same shape as ``low``.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``low`` and ``high`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default floating-point data type. Default ``None``
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized uniform distribution.

        Examples
        --------
        >>> x = ivy.array([[9.8, 3.4], [5.8, 7.2]])
        >>> x.random_uniform(10.2)
        ivy.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x = ivy.array([8.7, 9.3])
        >>> y = ivy.array([12.8, 14.5])
        >>> z = ivy.ones((2,))
        >>> x.random_uniform(y, device='cpu', dtype='float64', out=z)
        ivy.array([12.1, 14. ])
        """
        return ivy.random_normal(
            self._data,
            std,
            shape,
            device=device,
            dtype=dtype,
            out=out,
        )
