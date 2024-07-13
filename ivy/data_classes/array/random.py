# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithRandom(abc.ABC):
    def random_uniform(
        self: ivy.Array,
        /,
        *,
        high: Union[float, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Array, ivy.Shape, ivy.NativeShape]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.random_uniform. This method
        simply wraps the function, and so the docstring for ivy.random_uniform
        also applies to this method with minimal changes.

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
        seed
            A python integer. Used to create a random seed distribution
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
        >>> x.random_uniform(high=10.2)
        ivy.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x.random_uniform(high=10.2, device='cpu')
        ivy.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x.random_uniform(high=14.2, dtype='float16')
        ivy.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x.random_uniform(high=10.8, device='cpu', dtype='float64')
        ivy.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> z = ivy.ones((2,2))
        >>> x.random_uniform(high=11.2, device='cpu', dtype='float64', out=z)
        ivy.array([[10.1 ,  6.53],
                   [ 7.94,  8.85]])

        >>> x = ivy.array([8.7, 9.3])
        >>> y = ivy.array([12.8, 14.5])
        >>> x.random_uniform(y)
        ivy.array([12.1, 14. ])

        >>> x.random_uniform(high=y, device='cpu')
        ivy.array([12.1, 14. ])

        >>> x.random_uniform(high=y, dtype='float16')
        ivy.array([12.1, 14. ])

        >>> x.random_uniform(high=y, device='cpu', dtype='float64')
        ivy.array([12.1, 14. ])

        >>> z = ivy.ones((2,))
        >>> x.random_uniform(high=y, device='cpu', dtype='float64', out=z)
        ivy.array([12.1, 14. ])
        """
        return ivy.random_uniform(
            low=self._data,
            high=high,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def random_normal(
        self: ivy.Array,
        /,
        *,
        std: Union[float, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.random_normal. This method
        simply wraps the function, and so the docstring for ivy.random_normal
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The mean of the normal distribution to sample from. Default is ``0.0``.
        std
            The standard deviation of the normal distribution to sample from.
            Must be non-negative. Default is ``1.0``.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``mean`` and ``std`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized normal distribution.

        Examples
        --------
        >>> x = ivy.array([[9.8, 3.4], [5.8, 7.2]])
        >>> x.random_normal(std=10.2)
        ivy.array([[19.   , -6.44 ],
                   [ 5.72 ,  0.235]])

        >>> x.random_normal(std=10.2, device='cpu')
        ivy.array([[18.7 , 25.2 ],
                   [27.5 , -3.22]])

        >>> x.random_normal(std=14.2, dtype='float16')
        ivy.array([[26.6 , 12.1 ],
                   [ 4.56,  5.49]])

        >>> x.random_normal(std=10.8, device='cpu', dtype='float64')
        ivy.array([[ 1.02, -1.39],
                   [14.2 , -1.  ]])

        >>> z = ivy.ones((2,2))
        >>> x.random_normal(std=11.2, device='cpu', dtype='float64', out=z)
        ivy.array([[ 7.72, -8.32],
                   [ 4.95, 15.8 ]])

        >>> x = ivy.array([8.7, 9.3])
        >>> y = ivy.array([12.8, 14.5])
        >>> x.random_normal(std=y)
        ivy.array([-10.8,  12.1])

        >>> x.random_normal(std=y, device='cpu')
        ivy.array([ 13. , -26.9])

        >>> x.random_normal(std=y, dtype='float16')
        ivy.array([14.3  , -0.807])

        >>> x.random_normal(std=y, device='cpu', dtype='float64')
        ivy.array([21.3 ,  3.85])

        >>> z = ivy.ones((2,))
        >>> x.random_normal(std=y, device='cpu', dtype='float64', out=z)
        ivy.array([ 4.32, 42.2 ])
        """
        return ivy.random_normal(
            mean=self._data,
            std=std,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def multinomial(
        self: ivy.Array,
        population_size: int,
        num_samples: int,
        /,
        *,
        batch_size: int = 1,
        replace: bool = True,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.multinomial. This method
        simply wraps the function, and so the docstring for ivy.multinomial
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The unnormalized probabilities for all elements in population,
            default is uniform *[batch_shape, population_size]*
        population_size
            The size of the population from which to draw samples.
        num_samples
            Number of independent samples to draw from the population.
        batch_size
            Number of tensors to generate. Default is 1.
        replace
            Whether to replace samples once they've been drawn. Default is ``True``.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None)
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized normal distribution.
        """
        return ivy.multinomial(
            population_size,
            num_samples,
            batch_size=batch_size,
            probs=self._data,
            replace=replace,
            device=device,
            seed=seed,
            out=out,
        )

    def randint(
        self: ivy.Array,
        high: Union[int, ivy.Array, ivy.NativeArray],
        /,
        *,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.randint. This method simply
        wraps the function, and so the docstring for ivy.randint also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Lowest integer that can be drawn from the distribution.
        high
            One above the highest integer that can be drawn from the distribution.
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
             type will be the default integer data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Returns an array with the given shape filled with integers from
            the uniform distribution in the “half-open” interval [low, high)

        Examples
        --------
        >>> x = ivy.array([[1, 2], [0, 5]])
        >>> x.randint(10)
        ivy.array([[1, 5],
                   [9, 7]])

        >>> x.randint(8, device='cpu')
        ivy.array([[6, 5],
                   [0, 5]])

        >>> x.randint(9, dtype='int8')
        ivy.array([[1, 2],
                   [7, 7]])

        >>> x.randint(14, device='cpu', dtype='int16')
        ivy.array([[6, 5],
                   [0, 5]])

        >>> z = ivy.ones((2,2))
        >>> x.randint(16, device='cpu', dtype='int64', out=z)
        ivy.array([[1, 2],
                   [7, 7]])

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([23, 25, 98])
        >>> x.randint(y)
        ivy.array([ 5, 14, 18])

        >>> x.randint(y, device='cpu')
        ivy.array([20, 13, 46])

        >>> x.randint(y, dtype='int32')
        ivy.array([ 9, 18, 33])

        >>> x.randint(y, device='cpu', dtype='int16')
        ivy.array([ 9, 20, 85])

        >>> z = ivy.ones((3,))
        >>> x.randint(y, device='cpu', dtype='int64', out=z)
        ivy.array([20, 13, 46])
        """
        return ivy.randint(
            self._data,
            high,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def shuffle(
        self: ivy.Array,
        axis: Optional[int] = 0,
        /,
        *,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.shuffle. This method simply
        wraps the function, and so the docstring for ivy.shuffle also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array. Should have a numeric data type.
        axis
            The axis which x is shuffled along. Default is 0.
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array object, shuffled along the first dimension.

        Examples
        --------
        >>> x = ivy.array([5, 2, 9])
        >>> y = x.shuffle()
        >>> print(y)
        ivy.array([2, 5, 9])
        """
        return ivy.shuffle(self, axis, seed=seed, out=out)
