# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithRandom(ContainerBase):
    @staticmethod
    def static_random_uniform(
        low: Union[float, ivy.Container, ivy.Array, ivy.NativeArray] = 0.0,
        high: Union[float, ivy.Container, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.random_uniform. This method
        simply wraps the function, and so the docstring for ivy.random_uniform also
        applies to this method with minimal changes.

        Parameters
        ----------
        low
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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        With :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                              b=ivy.array([[0.9,2.4],[7.6,5.4]]))
        >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                              b=ivy.array([[4.3,5.6],[23.4,54.3]]))
        >>> ivy.Container.static_random_uniform(x, y, device='cpu', dtype='float64')
        {
            a: ivy.array([[10.8, 23.7],
                          [17., 16.6]]),
            b: ivy.array([[2.35, 3.69],
                          [17.4, 48.]])
        }

        With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

        >>> x = ivy.array([-1.0,-9.0,-3.4])
        >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, 0.2]))
        >>> ivy.Container.static_random_uniform(x, y)
        {
            a: ivy.array([0.481, -8.03, -2.74]),
            b: ivy.array([0.0999, -7.38, -1.29])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "random_uniform",
            low,
            high,
            shape,
            device=device,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def random_uniform(
        self: ivy.Container,
        high: Union[float, ivy.Container, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.random_uniform. This method
        simply wraps the function, and so the docstring for ivy.random_uniform also
        applies to this method with minimal changes.

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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        >>> x = ivy.Container(a=ivy.array([7.5,6.7,0.9]), b=ivy.array([8.7,9.8,4.5]))
        >>> x.random_uniform(17.4)
        {
            a: ivy.array([11.2, 10.5, 13.1]),
            b: ivy.array([11.2, 11.9, 6.01])
        }

        >>> y = ivy.Container(a=10.4, b=17.4)
        >>> x.random_uniform(y, device='cpu')
        {
            a: ivy.array([8.55, 10.1, 4.08]),
            b: ivy.array([9.45, 9.9, 8.6])
        }

        >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                              b=ivy.array([[0.9,2.4],[7.6,5.4]]))
        >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                              b=ivy.array([[4.3,5.6],[23.4,54.3]]))
        >>> z = ivy.Container(a=ivy.zeros((2,2)), b=ivy.ones((2,2)))
        >>> x.random_uniform(y, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([[10.4, 29.8],
                          [12.1, 3.9]]),
            b: ivy.array([[3.79, 5.4],
                          [16.2, 31.7]])
        }
        """
        return self.static_random_uniform(
            self,
            high,
            shape,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_random_normal(
        mean: Union[float, ivy.Container, ivy.Array, ivy.NativeArray] = 0.0,
        std: Union[float, ivy.Container, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.random_normal. This method
        simply wraps the function, and so the docstring for ivy.random_normal
        also applies to this method with minimal changes.

        Parameters
        ----------
        mean
            The mean of the normal distribution to sample from. Default is ``0.0``.
        std
            The standard deviation of the normal distribution to sample from.
            Must be non-negative. Default is ``1.0``.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``mean`` and ``std`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
            Drawn samples from the parameterized normal distribution.

        Examples (to update)
        --------
        With :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                              b=ivy.array([[0.9,2.4],[7.6,5.4]]))
        >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                              b=ivy.array([[4.3,5.6],[23.4,54.3]]))
        >>> ivy.Container.static_random_normal(x, y, device='cpu', dtype='float64')
        {
            a: ivy.array([[-4.11, 0.651],
                          [19.3, -30.4]]),
            b: ivy.array([[1.15, 3.39],
                          [-9.35, -13.9]])
        }

        With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

        >>> x = ivy.array([-1.0,-9.0,-3.4])
        >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, 0.2]))
        >>> ivy.Container.static_random_normal(x, y)
        {
            a: ivy.array([-0.651, -9.25, -3.54]),
            b: ivy.array([0.464, -8.51, -3.75])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "random_normal",
            mean,
            std,
            shape,
            device=device,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def random_normal(
        self: ivy.Container,
        std: Union[float, ivy.Container, ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.random_normal. This method
        simply wraps the function, and so the docstring for ivy.random_normal also
        applies to this method with minimal changes.

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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
            Drawn samples from the parameterized normal distribution.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([7.5,6.7,0.9]), b=ivy.array([8.7,9.8,4.5]))
        >>> x.random_normal(17.4)
        {
            a: ivy.array([8.79, 27.2, 7.01]),
            b: ivy.array([-18.6, 11.4, 15.])
        }

        >>> y = ivy.Container(a=10.4, b=17.4)
        >>> x.random_normal(y, device='cpu')
        {
            a: ivy.array([11.8, 7.45, 9.95]),
            b: ivy.array([-15.1, 29., 30.2])
        }

        >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                              b=ivy.array([[0.9,2.4],[7.6,5.4]]))
        >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                              b=ivy.array([[4.3,5.6],[23.4,54.3]]))
        >>> z = ivy.Container(a=ivy.zeros((2,2)), b=ivy.ones((2,2)))
        >>> x.random_normal(y, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([[-12.7, 72.7],
                          [12.8, -0.0762]]),
            b: ivy.array([[-6.56, -5.12], 
                          [12.8, 13.2]])
        }
        """
        return self.static_random_normal(
            self,
            std,
            shape,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            dtype=dtype,
            out=out,
        )

    # randint
    @staticmethod
    def static_randint(
        low: Union[int, ivy.Container] = 0.0,
        high: Union[int, ivy.Container] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.randint. This method simply wraps the
        function, and so the docstring for ivy.randint also applies to this method
        with minimal changes.

        Parameters
        ----------
        low
            Lowest integer that can be drawn from the distribution.
        high
            One above the highest integer that can be drawn from the distribution.
        shape
            a Sequence defining the shape of the output array.
        device
            device on which to create the array. 'cuda:0',
            'cuda:1', 'cpu' etc. (Default value = None).
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
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
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=10), high=20, shape=2)
        >>> print(x)
        {
            a: ivy.array([10, 15]),
            b: ivy.array([16, 12])
        }

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=4), high=15, shape=(3,2))
        >>> print(x)
        {
            a: ivy.array([[12, 3],
                         [5, 7],
                         [7, 2]]),
            b: ivy.array([[8, 10],
                         [9, 6],
                         [6, 7]])
        }

        >>> x = ivy.Container.randint(low=ivy.Container(a=5,b=20,c=40),\
                                      high=100,\
                                      shape=3,\
                                      device='gpu:1')
        >>> print(x)
        {
            a: ivy.array([90, 87, 62]),
            b: ivy.array([52, 95, 37]),
            c: ivy.array([95, 90, 42])
        }

        >>> x = ivy.Container(a=1,b=2)
        >>> y = ivy.Container.randint(low=ivy.Container(a=3,b=5,c=10,d=7),\
                                      high=14,\
                                      shape=5,\
                                      out=x)
        >>> print(x)
        {
            a: ivy.array([4, 10, 13, 3, 3]),
            b: ivy.array([12, 11, 11, 12, 5]),
            c: ivy.array([10, 13, 11, 13, 12]),
            d: ivy.array([12, 7, 8, 11, 8])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=10),\
                                      high=ivy.Container(a=5, b= 15, c=2),\
                                      shape=2)
        >>> print(x)
        {
            a: ivy.array([1, 2]),
            b: ivy.array([14, 10])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "randint",
            low,
            high,
            shape,
            device=device,
            key_chains=key_chains,
            out=out,
        )
