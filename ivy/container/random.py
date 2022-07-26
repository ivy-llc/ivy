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

        >>> x.random_uniform(10.2, device='cpu')
        {
            a: ivy.array([8.55, 10.1, 4.08]),
            b: ivy.array([9.45, 9.9, 8.6])
        }

        >>> x.random_uniform(14.2, dtype='float16')
        {
            a: ivy.array([12.4, 11.7, 7.25]),
            b: ivy.array([11.8, 11.8, 4.96])
        }

        >>> x.random_uniform(10.8, device='cpu', dtype='float64')
        {
            a: ivy.array([8.86, 9.24, 6.43]),
            b: ivy.array([8.95, 10.1, 8.51])
        }

        >>> z = ivy.Container(a=ivy.zeros((3,)), b=ivy.ones((3,)))
        >>> x.random_uniform(11.2, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([9.6, 8.24, 3.67]),
            b: ivy.array([9.29, 11.2, 9.84])
        }

        >>> y = ivy.Container(a=10.4, b=17.4)
        >>> x.random_uniform(y)
        {
            a: ivy.array([8.24, 9.22, 1.52]),
            b: ivy.array([16.5, 13.4, 17.3])
        }

        >>> x.random_uniform(y, device='cpu')
        {
            a: ivy.array([8.55, 10.1, 4.08]),
            b: ivy.array([9.45, 9.9, 8.6])
        }

        >>> x.random_uniform(y, dtype='float16')
        {
            a: ivy.array([12.4, 11.7, 7.25]),
            b: ivy.array([11.8, 11.8, 4.96])
        }

        >>> x.random_uniform(y, device='cpu', dtype='float64')
        {
            a: ivy.array([8.86, 9.24, 6.43]),
            b: ivy.array([8.95, 10.1, 8.51])
        }

        >>> z = ivy.Container(a=ivy.zeros((3,)), b=ivy.ones((3,)))
        >>> x.random_uniform(y, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([9.6, 8.24, 3.67]),
            b: ivy.array([9.29, 11.2, 9.84])
        }

        >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                              b=ivy.array([[0.9,2.4],[7.6,5.4]]))
        >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                              b=ivy.array([[4.3,5.6],[23.4,54.3]]))
        >>> x.random_uniform(y)
        {
            a: ivy.array([[10.4, 17.],
                          [9.81, 10.9]]),
            b: ivy.array([[3.6, 4.31],
                          [18.8, 54.2]])
        }

        >>> x.random_uniform(y, device='cpu')
        {
            a: ivy.array([[10.1, 7.93],
                          [7.98, 6.]]),
            b: ivy.array([[4.28, 4.65],
                          [13.9, 28.9]])
        }

        >>> x.random_uniform(y, dtype='float16')
        {
            a: ivy.array([[10.6, 28.],
                          [16.4, 4.92]]),
            b: ivy.array([[3.61, 4.82],
                          [12.6, 10.2]])
        }

        >>> x.random_uniform(y, device='cpu', dtype='float64')
        {
            a: ivy.array([[10.7, 28.4],
                          [9.29, 17.4]]),
            b: ivy.array([[1.88, 4.94],
                          [17., 9.68]])
        }

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

        Examples
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
        >>> x = ivy.Container(a=ivy.array([7.5,6.7,0.9]), \
                              b=ivy.array([8.7,9.8,4.5]))
        >>> x.random_normal(17.4)
        {
            a: ivy.array([11.9, -22.9, -24.8]),
            b: ivy.array([44.3, -21.6, 2.03])
        }

        >>> x.random_normal(10.2, device='cpu')
        {
            a: ivy.array([7.82, 6.21, -0.431]),
            b: ivy.array([13.8, 9.9, 7.64])
        }

        >>> x.random_normal(14.2, dtype='float16')
        {
            a: ivy.array([-18.3, -3.42, 9.55]),
            b: ivy.array([-1.31, 7.68, -6.93])
        }

        >>> x.random_normal(10.8, device='cpu', dtype='float64')
        {
            a: ivy.array([13.4, -3.14, 10.7]),
            b: ivy.array([11.7, 4.85, 5.83])
        }

        >>> z = ivy.Container(a=ivy.zeros((3,)), b=ivy.ones((3,)))
        >>> x.random_normal(11.2, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([-6.84, 0.274, 14.2]),
            b: ivy.array([29.1, 7.19, 3.])
        }

        >>> y = ivy.Container(a=10.4, b=17.4)
        >>> x.random_normal(y)
        {
            a: ivy.array([-9.5, 8.54, -9.13]),
            b: ivy.array([-24.5, 18.9, 11.])
        }

        >>> x.random_normal(y, device='cpu')
        {
            a: ivy.array([8.47, 8.23, 8.69]),
            b: ivy.array([10.7, 16.2, 16.1])
        }

        >>> x.random_normal(y, dtype='float16')
        {
            a: ivy.array([8.22, -15.9, 10.4]),
            b: ivy.array([19.9, 11.5, -2.15])
        }

        >>> x.random_normal(y, device='cpu', dtype='float64')
        {
            a: ivy.array([19.6, -4.08, 6.09]),
            b: ivy.array([-23.9, 6.86, 17.6])
        }

        >>> z = ivy.Container(a=ivy.zeros((3,)), b=ivy.ones((3,)))
        >>> x.random_normal(y, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([14.7, 8.99, 8.46]),
            b: ivy.array([22.9, -5.97, -1.28])
        }

        >>> x = ivy.Container(a=ivy.array([[9.8,7.6],[6.5,2.3]]), \
                              b=ivy.array([[0.9,2.4],[7.6,5.4]]))
        >>> y = ivy.Container(a=ivy.array([[10.9,32.4],[18.7,19.6]]), \
                              b=ivy.array([[4.3,5.6],[23.4,54.3]]))
        >>> x.random_normal(y)
        {
            a: ivy.array([[10.6, 7.89],
                          [9.39, 19.4]]),
            b: ivy.array([[3.76, 4.68],
                          [17.7, 24.]])
        }

        >>> x.random_normal(y, device='cpu')
        {
            a: ivy.array([[30.9, 24.6],
                          [29.9, -25.3]]),
            b: ivy.array([[8.02, 1.92],
                          [-5.34, -54.1]])
        }

        >>> x.random_normal(y, dtype='float16')
        {
            a: ivy.array([[7.82, -35.],
                          [11.7, 0.696]]),
            b: ivy.array([[-4.07, -2.91],
                          [19.2, 46.8]])
        }

        >>> x.random_normal(y, device='cpu', dtype='float64')
        {
            a: ivy.array([[25.4, 28.3],
                          [19.6, -9.83]]),
            b: ivy.array([[2.95, 2.48],
                          [-30.8, -40.1]])
        }

        >>> z = ivy.Container(a=ivy.zeros((2,2)), b=ivy.ones((2,2)))
        >>> x.random_normal(y, device='cpu', dtype='float64', out=z)
        {
            a: ivy.array([[2.8, -45.6],
                          [-10.4, 0.65]]),
            b: ivy.array([[3.8, 1.43],
                          [23., 29.4]])
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

    @staticmethod
    def static_randint(
        low: Union[int, ivy.Container, ivy.Array, ivy.NativeArray],
        high: Union[int, ivy.Container, ivy.Array, ivy.NativeArray],
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
        """ivy.Container static method variant of ivy.randint. This method
        simply wraps the function, and so the docstring for ivy.randint also
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
            "randint",
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

    def randint(
        self: ivy.Container,
        high: Union[int, ivy.Container, ivy.Array, ivy.NativeArray],
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
        """ivy.Container instance method variant of ivy.randint. This method
        simply wraps the function, and so the docstring for ivy.randint also
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
        return self.static_randint(
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

