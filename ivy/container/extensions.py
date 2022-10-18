# global
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Literal,
    Iterable,
    Sequence,
)
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithExtensions(ContainerBase):
    @staticmethod
    def static_sinc(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the sinc of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
        ...                   b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = ivy.Container.static_sinc(x)
        >>> print(y)
        {
            a: ivy.array([0.636, -0.212, 0.127]),
            b: ivy.array([-0.090, 0.070, -0.057])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sinc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinc(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the sinc of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
        ...                   b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = x.sinc()
        >>> print(y)
        {
            a: ivy.array([0.637,-0.212,0.127]),
            b: ivy.array([-0.0909,0.0707,-0.0579])
        }
        """
        return self.static_sinc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_flatten(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        start_dim: Optional[int] = 0,
        end_dim: Optional[int] = -1,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.flatten. This method simply wraps the
        function, and so the docstring for ivy.flatten also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container to flatten at leaves.
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            Container with arrays flattened at leaves.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ...                   b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> ivy.flatten(x)
        [{
            a: ivy.array([1, 2, 3, 4, 5, 6, 7, 8])
            b: ivy.array([9, 10, 11, 12, 13, 14, 15, 16])
        }]
        """
        return ContainerBase.multi_map_in_static_method(
            "flatten",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            start_dim=start_dim,
            end_dim=end_dim,
            out=out,
        )

    def flatten(
        self: ivy.Container,
        *,
        start_dim: Optional[int] = 0,
        end_dim: Optional[int] = -1,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.flatten. This method simply
        wraps the function, and so the docstring for ivy.flatten also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input container to flatten at leaves.
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            Container with arrays flattened at leaves.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ...                   b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> ivy.flatten(x)
        [{
            a: ivy.array([1, 2, 3, 4, 5, 6, 7, 8])
            b: ivy.array([9, 10, 11, 12, 13, 14, 15, 16])
        }]
        """
        return self.static_flatten(self, start_dim=start_dim, end_dim=end_dim, out=out)

    @staticmethod
    def static_lcm(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),
        ...                  b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),
        ...                  b=ivy.array([32, 40, 25, 13]))
        >>> ivy.Container.lcm(x1, x2)
        {
            a: ivy.array([10, 21, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "lcm",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lcm(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),
        ...                  b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),
        ...                  b=ivy.array([32, 40, 25, 13]))
        >>> x1.lcm(x2)
        {
            a: ivy.array([10, 21, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return self.static_lcm(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_hann_window(
        window_length: Union[int, ivy.Container],
        periodic: Optional[bool] = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hann_window. This method simply wraps
        the function, and so the docstring for ivy.hann_window also applies to this
        method with minimal changes.

        Parameters
        ----------
        window_length
            container including multiple window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that contains the Hann windows.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_hann(x)
        {
            a: ivy.array([0.0000, 0.7500, 0.7500])
            b: ivy.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "hann_window",
            window_length,
            periodic,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hann_window(
        self: ivy.Container,
        periodic: Optional[bool] = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.hann_window. This method simply
        wraps the function, and so the docstring for ivy.hann_window also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container containing the Hann windows.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.hann_window(x)
        {
            a: ivy.array([0.0000, 0.7500, 0.7500])
            b: ivy.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455])
        }
        """
        return self.static_hann_window(self, periodic, dtype, out=out)

    @staticmethod
    def static_max_pool2d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.max_pool2dd. This method simply
        wraps the function, and so the docstring for ivy.max_pool2d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d_in]*.
        kernel
            The size of the window to take a max over.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12).reshape((2, 1, 3, 2))
        >>> b = ivy.arange(48).reshape((2, 4, 3, 2))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(ivy.Container.static_max_pool2d(x, (2, 2), (1, 1), "SAME"))
        {
            a: (<class ivy.array.array.Array> shape=[2, 1, 3, 2]),
            b: (<class ivy.array.array.Array> shape=[2, 4, 3, 2])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "max_pool2d",
            x,
            kernel,
            strides,
            padding,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def max_pool2d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of `ivy.max_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool2d` also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d_in]*.
        kernel
            The size of the window to take a max over.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12).reshape((2, 1, 3, 2))
        >>> b = ivy.arange(48).reshape((2, 4, 3, 2))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(x.max_pool2d(2, 2), (1, 1), "SAME"))
        {
            a: (<class ivy.array.array.Array> shape=[2, 1, 3, 2]),
            b: (<class ivy.array.array.Array> shape=[2, 4, 3, 2])
        }
        """
        return self.static_max_pool2d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_kaiser_window(
        window_length: Union[int, ivy.Container],
        periodic: bool = True,
        beta: float = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.kaiser_window. This method
        simply wraps the function, and so the docstring for ivy.kaiser_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        window_length
            input container including window lenghts.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_kaiser_window(x, True, 5)
        {
            a: ivy.array([0.2049, 0.8712, 0.8712]),
            a: ivy.array([0.0367, 0.7753, 0.7753]),
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "kaiser_window",
            window_length,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def kaiser_window(
        self: ivy.Container,
        periodic: bool = True,
        beta: float = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.kaiser_window. This method
        simply wraps the function, and so the docstring for ivy.kaiser_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container including window lenghts.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_kaiser_window(x, True, 5)
        {
            a: ivy.array([0.2049, 0.8712, 0.8712]),
            a: ivy.array([0.0367, 0.7753, 0.7753]),
        }
        """
        return self.static_kaiser_window(
            self,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_moveaxis(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.moveaxis. This method simply wraps
        the function, and so the docstring for ivy.moveaxis also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            The container with the arrays whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with moved axes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.zeros((3, 4, 5)), b=ivy.zeros((2,7,6)))
        >>> ivy.Container.static_moveaxis(x, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "moveaxis",
            a,
            source,
            destination,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def moveaxis(
        self: ivy.Container,
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.moveaxis. This method simply
        wraps the function, and so the docstring for ivy.flatten also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            The container with the arrays whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with moved axes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.zeros((3, 4, 5)), b=ivy.zeros((2,7,6)))
        >>> x.moveaxis(, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return self.static_moveaxis(self, source, destination, out=out)

    @staticmethod
    def static_pad(
        x: ivy.Container,
        /,
        *,
        pad_width: Union[Iterable[Tuple[int]], int],
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return ContainerBase.multi_map_in_static_method(
            "pad",
            x,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def pad(
        self: ivy.Container,
        /,
        *,
        pad_width: Union[Iterable[Tuple[int]], int],
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return self.static_pad(
            self,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_heaviside(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.heaviside. This method simply wraps
        the function, and so the docstring for ivy.heaviside also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            input container including the arrays.
        x2
            values to use where the array is zero.
        out
            optional output container array, for writing the result to.

        Returns
        -------
        ret
            output container with element-wise Heaviside step function of each array.

        Examples
        --------
        With :class:`ivy.Array` input:
        >>> x1 = ivy.Container(a=ivy.array([-1.5, 0, 2.0]), b=ivy.array([3.0, 5.0])
        >>> x2 = ivy.Container(a=0.5, b=[1.0, 2.0])
        >>> ivy.Container.static_heaviside(x1, x2)
        {
            a: ivy.array([ 0. ,  0.5,  1. ])
            b: ivy.array([1.0, 1.0])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "heaviside",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def heaviside(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.heaviside. This method simply
        wraps the function, and so the docstring for ivy.heaviside also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input container including the arrays.
        x2
            values to use where the array is zero.
        out
            optional output container array, for writing the result to.

        Returns
        -------
        ret
            output container with element-wise Heaviside step function of each array.

        Examples
        --------
        With :class:`ivy.Array` input:
        >>> x1 = ivy.Container(a=ivy.array([-1.5, 0, 2.0]), b=ivy.array([3.0, 5.0])
        >>> x2 = ivy.Container(a=0.5, b=[1.0, 2.0])
        >>> x1.heaviside(x2)
        {
            a: ivy.array([ 0. ,  0.5,  1. ])
            b: ivy.array([1.0, 1.0])
        }
        """
        return self.static_heaviside(self, x2, out=out)

    @staticmethod
    def static_median(
        input: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.median. This method simply wraps
        the function, and so the docstring for ivy.median also applies to this method
        with minimal changes.

        Parameters
        ----------
        input
            Input container including arrays.
        axis
            Axis or axes along which the medians are computed. The default is to compute
            the median along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The median of the array elements.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.zeros((3, 4, 5)), b=ivy.zeros((2,7,6)))
        >>> ivy.Container.static_moveaxis(x, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "median",
            input,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def median(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.median. This method simply
        wraps the function, and so the docstring for ivy.median also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        axis
            Axis or axes along which the medians are computed. The default is to compute
            the median along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The median of the array elements.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(
        >>>     a=ivy.array([[10, 7, 4], [3, 2, 1]]),
        >>>     b=ivy.array([[1, 4, 2], [8, 7, 0]])
        >>> )
        >>> x.median(axis=0)
        {
            a: ivy.array([6.5, 4.5, 2.5]),
            b: ivy.array([4.5, 5.5, 1.])
        }
        """
        return self.static_median(self, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def static_flipud(
        m: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.flipud. This method simply wraps
        the function, and so the docstring for ivy.flipud also applies to this method
        with minimal changes.

        Parameters
        ----------
        m
            the container with arrays to be flipped.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the input container's array
            with elements order reversed along axis 0.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> m = ivy.Container(a=ivy.diag([1, 2, 3]), b=ivy.arange(4))
        >>> ivy.Container.static_flipud(m)
        {
            a: ivy.array(
                [[ 0.,  0.,  3.],
                 [ 0.,  2.,  0.],
                 [ 1.,  0.,  0.]]
            )
            b: ivy.array([3, 2, 1, 0])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "flipud",
            m,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def flipud(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.flipud. This method simply
        wraps the function, and so the docstring for ivy.flipud also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            the container with arrays to be flipped.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the input container's array
            with elements order reversed along axis 0.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> m = ivy.Container(a=ivy.diag([1, 2, 3]), b=ivy.arange(4))
        >>> m.flipud()
        {
            a: ivy.array(
                [[ 0.,  0.,  3.],
                 [ 0.,  2.,  0.],
                 [ 1.,  0.,  0.]]
            )
            b: ivy.array([3, 2, 1, 0])
        }
        """
        return self.static_flipud(self, out=out)

    @staticmethod
    def static_fmod(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmod. This method simply wraps
        the function, and so the docstring for ivy.fmod also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> ivy.Container.static_fmod(x1, x2)
        {
            a: ivy.array([ 0,  3,  0])
            b: ivy.array([ nan,  nan,  nan])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "fmod",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmod(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.fmod. This method simply
        wraps the function, and so the docstring for ivy.fmod also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> x1.fmod(x2)
        {
            a: ivy.array([ 0,  3,  0])
            b: ivy.array([ nan,  nan,  nan])
        }
        """
        return self.static_fmod(self, x2, out=out)

    def fmax(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.fmax. This method simply
        wraps the function, and so the docstring for ivy.fmax also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> x1.fmax(x2)
        {
            a: ivy.array([ 2.,  5.,  4.])
            b: ivy.array([ 0,  0,  nan])
        }
        """
        return self.static_fmax(self, x2, out=out)
