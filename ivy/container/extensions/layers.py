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
    Any,
)
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithLayersExtensions(ContainerBase):
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
    def static_max_pool1d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.max_pool1d. This method simply
        wraps the function, and so the docstring for ivy.max_pool1d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Container of input images *[batch_size, w, d_in]*.
        kernel
            Size of the kernel i.e., the sliding window for each
            dimension of input. *[w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list
            indicating the per-dimension paddings.
        data_format
            NWC" or "NCW". Defaults to "NWC".
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12.).reshape((2,2,3))
        >>> b = ivy.arange(24.).reshape((2,3,4))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(ivy.Container.static_max_pool1d(x,2, 2, "VALID"))
        {
            a: ivy.array([[[3., 4., 5.]],
                          [[9., 10., 11.]]]),
            b: ivy.array([[[4., 5., 6., 7.]],
                          [[16., 17., 18., 19.]]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "max_pool1d",
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

    def max_pool1d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of `ivy.max_pool1d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool1d` also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Container of input images *[batch_size, w, d_in]*.
        kernel
            Size of the kernel i.e., the sliding window for each
            dimension of input. *[w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list
            indicating the per-dimension paddings.
        data_format
            NWC" or "NCW". Defaults to "NWC".
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12.).reshape((2,2,3))
        >>> b = ivy.arange(24.).reshape((2,3,4))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(x.max_pool1d(2, 2, "VALID"))
        {
            a: ivy.array([[[3., 4., 5.]],
                          [[9., 10., 11.]]]),
            b: ivy.array([[[4., 5., 6., 7.]],
                          [[16., 17., 18., 19.]]])
        }
        """
        return self.static_max_pool1d(
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
    def static_pad(
        input: ivy.Container,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
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
        **kwargs: Optional[Any],
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return ContainerBase.multi_map_in_static_method(
            "pad",
            input,
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
            **kwargs,
        )

    def pad(
        self: ivy.Container,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
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
        **kwargs: Optional[Any],
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
            **kwargs,
        )

    @staticmethod
    def static_kaiser_bessel_derived_window(
        x: Union[int, ivy.Array, ivy.NativeArray, ivy.Container],
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
        ivy.Container static method variant of ivy.kaiser_bessel_derived_window.
        This method simply wraps the function, and so the docstring for
        ivy.kaiser_bessel_derived_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
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
            The container that includes the Kaiser Bessel Derived windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_kaiser_bessel_derived_window(x, True, 5)
        {
            a: ivy.array([0.70710677, 0.70710677]),
            b: ivy.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208]),
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "kaiser_bessel_derived_window",
            x,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def kaiser_bessel_derived_window(
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
        """ivy.Container instance method variant of ivy.kaiser_bessel_derived_window.
        This method simply wraps the function, and so the docstring for
        ivy.kaiser_bessel_derived_window also applies to this method with
        minimal changes.

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
            The container that includes the Kaiser Bessel Derived windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5))
        >>> x.kaiser_bessel_derived_window(True, 5)
        {
            a: ivy.array([0.70710677, 0.70710677]),
            b: ivy.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208]),
        }
        """
        return self.static_kaiser_bessel_derived_window(
            self, periodic, beta, dtype=dtype, out=out
        )

    @staticmethod
    def static_hamming_window(
        x: Union[int, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        periodic: Optional[bool] = True,
        alpha: Optional[float] = 0.54,
        beta: Optional[float] = 0.46,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hamming_window.
        This method simply wraps the function, and so the docstring for
        ivy.hamming_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container including window lenghts.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        alpha
            The coefficient alpha in the hamming window equation
        beta
            The coefficient beta in the hamming window equation
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Hamming windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_hamming_window(x, periodic=True, alpha=0.2, beta=2)
        {
            a: ivy.array([-1.8000,  1.2000,  1.2000]),
            b: ivy.array([-1.8000, -0.4180,  1.8180,  1.8180, -0.4180])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "hamming_window",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            periodic=periodic,
            alpha=alpha,
            beta=beta,
            dtype=dtype,
            out=out,
        )

    def hamming_window(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        periodic: Optional[bool] = True,
        alpha: Optional[float] = 0.54,
        beta: Optional[float] = 0.46,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.hamming_window.
        This method simply wraps the function, and so the docstring for
        ivy.hamming_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container including window lenghts.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        alpha
            The coefficient alpha in the hamming window equation
        beta
            The coefficient beta in the hamming window equation
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Hamming windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5))
        >>> x.hamming_window(periodic=True, alpha=0.2, beta=2)
        {
            a: ivy.array([-1.8000,  1.2000,  1.2000]),
            b: ivy.array([-1.8000, -0.4180,  1.8180,  1.8180, -0.4180])
        }
        """
        return self.static_hamming_window(
            self, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, out=out
        )
