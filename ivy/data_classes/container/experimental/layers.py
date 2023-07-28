# global
from typing import Optional, Union, List, Dict, Tuple, Literal, Sequence

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithLayersExperimental(ContainerBase):
    @staticmethod
    def static_max_pool1d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
        /,
        *,
        data_format: str = "NWC",
        dilation: Union[int, Tuple[int]] = 1,
        ceil_mode: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.max_pool1d. This method simply wraps
        the function, and so the docstring for ivy.max_pool1d also applies to this
        method with minimal changes.

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
        return ContainerBase.cont_multi_map_in_function(
            "max_pool1d",
            x,
            kernel,
            strides,
            padding,
            data_format=data_format,
            dilation=dilation,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def max_pool1d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
        /,
        *,
        data_format: str = "NWC",
        dilation: Union[int, Tuple[int]] = 1,
        ceil_mode: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of `ivy.max_pool1d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool1d` also applies to
        this method with minimal changes.

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
            dilation=dilation,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_max_pool2d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.max_pool2dd. This method simply wraps
        the function, and so the docstring for ivy.max_pool2d also applies to this
        method with minimal changes.

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
        return ContainerBase.cont_multi_map_in_function(
            "max_pool2d",
            x,
            kernel,
            strides,
            padding,
            data_format=data_format,
            dilation=dilation,
            ceil_mode=ceil_mode,
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
        dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of `ivy.max_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.max_pool2d` also applies to
        this method with minimal changes.

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
        >>> print(x.max_pool2d((2, 2), (1, 1), "SAME"))
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
            dilation=dilation,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_max_pool3d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.max_pool3d. This method simply wraps
        the function, and so the docstring for ivy.max_pool3d also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        kernel
            Convolution filters *[d,h,w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12).reshape((1, 2, 1, 3, 2))
        >>> b = ivy.arange(48).reshape((2, 2, 2, 3, 2))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(ivy.Container.static_max_pool3d(x, 2, 1, "VALID"))
        {
            a: ivy.array([], shape=(1, 1, 0, 2, 2)),
            b: ivy.array([[[[[20, 21],
                             [22, 23]]]],
                       [[[[44, 45],
                             [46, 47]]]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "max_pool3d",
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

    def max_pool3d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.max_pool3d. This method simply wraps
        the function, and so the docstring for ivy.max_pool3d also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        kernel
            Convolution filters *[d,h,w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12).reshape((1, 2, 1, 3, 2))
        >>> b = ivy.arange(48).reshape((2, 2, 2, 3, 2))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(x.max_pool3d(2, 1, "VALID"))
        {
            a: ivy.array([], shape=(1, 1, 0, 2, 2)),
            b: ivy.array([[[[[20, 21],
                             [22, 23]]]],
                       [[[[44, 45],
                             [46, 47]]]]])
        }
        """
        return self.static_max_pool3d(
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
    def static_avg_pool1d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.avg_pool1d. This method simply wraps
        the function, and so the docstring for ivy.avg_pool1d also applies to this
        method with minimal changes.

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
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
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
        >>> print(ivy.Container.static_avg_pool1d(x,2, 2, "VALID"))
        {
            a: ivy.array([[[1.5, 2.5, 3.5]],
                          [[7.5, 8.5, 9.5]]]),
            b: ivy.array([[[2., 3., 4., 5.]],
                          [[14., 15., 16., 17.]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "avg_pool1d",
            x,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def avg_pool1d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int]],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of `ivy.avg_pool1d`. This method simply
        wraps the function, and so the docstring for `ivy.avg_pool1d` also applies to
        this method with minimal changes.

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
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
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
        >>> print(x.avg_pool1d(2, 2, "VALID"))
        {
            a: ivy.array([[[1.5, 2.5, 3.5]],
                          [[7.5, 8.5, 9.5]]]),
            b: ivy.array([[[2., 3., 4., 5.]],
                          [[14., 15., 16., 17.]]])
        }
        """
        return self.static_avg_pool1d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_avg_pool2d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        divisor_override: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.avg_pool2d. This method simply wraps
        the function, and so the docstring for ivy.avg_pool2d also applies to this
        method with minimal changes.

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
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If specified, it will be used as divisor,
            otherwise kernel_size will be used.
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
        >>> print(ivy.Container.static_avg_pool2d(x, (2, 2), (1, 1), "SAME"))
        {
            a: ivy.array([], shape=(2, 0, 2, 2)),
            b: (<class ivy.array.array.Array> shape=[2, 3, 2, 2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "avg_pool2d",
            x,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def avg_pool2d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        divisor_override: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of `ivy.avg_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.avg_pool2d` also applies to
        this method with minimal changes.

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
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If specified, it will be used as divisor,
            otherwise kernel_size will be used.
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
        >>> print(x.avg_pool2d((2, 2), (1, 1), "SAME"))
        {
            a: (<class ivy.array.array.Array> shape=[2, 1, 3, 2]),
            b: (<class ivy.array.array.Array> shape=[2, 4, 3, 2])
        }
        """
        return self.static_avg_pool2d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_avg_pool3d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        divisor_override: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.avg_pool3d. This method simply wraps
        the function, and so the docstring for ivy.avg_pool3d also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        kernel
            Convolution filters *[d,h,w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If specified, it will be used as the divisor, otherwise
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12).reshape((1, 2, 1, 3, 2))
        >>> b = ivy.arange(48).reshape((2, 2, 2, 3, 2))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(ivy.Container.static_avg_pool3d(x, 2, 1, "VALID"))
        {
            a: ivy.array([], shape=(1, 1, 0, 2, 2)),
            b: ivy.array([[[[[10., 11.],
                             [12., 13.]]]],
                       [[[[34., 35.],
                             [36., 37.]]]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "avg_pool3d",
            x,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def avg_pool3d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        count_include_pad: bool = False,
        ceil_mode: bool = False,
        divisor_override: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.avg_pool3d. This method simply wraps
        the function, and so the docstring for ivy.avg_pool3d also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        kernel
            Convolution filters *[d,h,w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            NDHWC" or "NCDHW". Defaults to "NDHWC".
        count_include_pad
            Whether to include padding in the averaging calculation.
        ceil_mode
            Whether to use ceil or floor for creating the output shape.
        divisor_override
            If specified, it will be used as the divisor, otherwise
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the pooling operation.

        Examples
        --------
        >>> a = ivy.arange(12).reshape((1, 2, 1, 3, 2))
        >>> b = ivy.arange(48).reshape((2, 2, 2, 3, 2))
        >>> x = ivy.Container({'a': a, 'b': b})
        >>> print(x.max_pool3d(2, 1, "VALID"))
        {
            a: ivy.array([], shape=(1, 1, 0, 2, 2)),
            b: ivy.array([[[[[20, 21],
                             [22, 23]]]],
                       [[[[44, 45],
                             [46, 47]]]]])
        }
        """
        return self.static_avg_pool3d(
            self,
            kernel,
            strides,
            padding,
            data_format=data_format,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_dct(
        x: ivy.Container,
        /,
        *,
        type: Literal[1, 2, 3, 4] = 2,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Optional[Literal["ortho"]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.dct. This method simply wraps the
        function, and so the docstring for ivy.dct also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Container with the input signals.
        type
            The type of the dct. Must be 1, 2, 3 or 4.
        n
            The lenght of the transform. If n is less than the input signal lenght,
            then x is truncated, if n is larger than x is zero-padded.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The transformed input.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
        ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
        >>> ivy.Container.static_dct(x, type=2, norm='ortho')
        {
            a: ivy.array([102., -51.5, 0., -5.39, 0., -1.61, 0.,
                        -0.406]),
            b: ivy.array([12.7, -6.44, 0., -0.673, 0., -0.201, 0.,
                        -0.0507])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([  8, 16,  24,  32,   40,   48,   56,   64]),
        ...                   b=ivy.array([11., 54, 23., 13., 255., 255., 132., 182.]))
        >>> n = ivy.Container(a=9, b=5)
        >>> type = ivy.Container(a=2, b=4)
        >>> norm = ivy.Container(a="ortho", b=None)
        >>> ivy.Container.static_dct(x, type=type, n=n, norm=norm)
        {
            a: ivy.array([96., -28.2, -31.9, 22.9, -26., 19.8, -17., 10.9,
                        -5.89]),
            b: ivy.array([242., -253., 286., -515., 467.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "dct",
            x,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def dct(
        self: ivy.Container,
        /,
        *,
        type: Literal[1, 2, 3, 4] = 2,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Optional[Literal["ortho"]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.dct. This method simply wraps the
        function, and so the docstring for ivy.dct also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Container with the input signals.
        type
            The type of the dct. Must be 1, 2, 3 or 4.
        n
            The lenght of the transform. If n is less than the input signal lenght,
            then x is truncated, if n is larger then x is zero-padded.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The transformed input.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
        ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
        >>> x.dct(type=2, norm='ortho')
        {
            a: ivy.array([102., -51.5, 0., -5.39, 0., -1.61, 0.,
                        -0.406]),
            b: ivy.array([12.7, -6.44, 0., -0.673, 0., -0.201, 0.,
                        -0.0507])
        }
        """
        return self.static_dct(
            self,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            out=out,
        )

    @staticmethod
    def static_idct(
        x: ivy.Container,
        /,
        *,
        type: Literal[1, 2, 3, 4] = 2,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Optional[Literal["ortho"]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.idct. This method simply wraps the
        function, and so the docstring for ivy.idct also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Container with the input signals.
        type
            The type of the dct. Must be 1, 2, 3 or 4.
        n
            The length of the transform. If n is less than the input signal length,
            then x is truncated, if n is larger than x is zero-padded.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The transformed input.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
        ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
        >>> ivy.Container.static_idct(x, type=2, norm='ortho')
        {
            a: ivy.array([79.49862671, -70.37691498, 30.00390816, -23.58938599,
                          13.92713165, -10.078475, 5.19664812, -1.95411837]),
            b: ivy.array([9.93732834, -8.79711437, 3.75048852, -2.94867325, 1.74089146,
                          -1.25980937, 0.64958102, -0.2442648])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([  8, 16,  24,  32,   40,   48,   56,   64]),
        ...                   b=ivy.array([11., 54, 23., 13., 255., 255., 132., 182.]))
        >>> n = ivy.Container(a=9, b=5)
        >>> type = ivy.Container(a=2, b=4)
        >>> norm = ivy.Container(a="ortho", b=None)
        >>> ivy.Container.static_idct(x, type=type, n=n, norm=norm)
        {
            a: ivy.array([86.29723358, -66.6950531, 9.93914509, 2.88008738,
                          -16.18951225, 18.06697273, -17.57439804, 11.68861485,
                          -4.41308832]),
            b: ivy.array([242.0700836, -253.2449036, 285.6711426, -514.501709,
                          467.4924011])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "idct",
            x,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def idct(
        self: ivy.Container,
        /,
        *,
        type: Literal[1, 2, 3, 4] = 2,
        n: Optional[int] = None,
        axis: int = -1,
        norm: Optional[Literal["ortho"]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.idct. This method simply wraps the
        function, and so the docstring for ivy.idct also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Container with the input signals.
        type
            The type of the idct. Must be 1, 2, 3 or 4.
        n
            The length of the transform. If n is less than the input signal length,
            then x is truncated, if n is larger then x is zero-padded.
        norm
            The type of normalization to be applied. Must be either None or "ortho".
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The transformed input.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
        ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
        >>> x.idct(type=2, norm='ortho')
        {
            a: ivy.array([79.49862671, -70.37691498, 30.00390816, -23.58938599,
                  13.92713165, -10.078475, 5.19664812, -1.95411837]),
            b: ivy.array([9.94, -8.79711437, 3.76, -2.94867325, 1.74089146,
                  -1.25980937, 0.64958102, -0.2442648])
        }
        """
        return self.static_idct(
            self,
            type=type,
            n=n,
            axis=axis,
            norm=norm,
            out=out,
        )

    @staticmethod
    def _static_fft(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        dim: int,
        /,
        *,
        norm: str = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fft. This method simply wraps the
        function, and so the docstring for ivy.fft also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Container containing input volumes *[...,d_in,...]*,
            where d_in indicates the dimension that needs FFT.
        dim
            The dimension along which to take the one dimensional FFT.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        n
            Optional argument indicating the sequence length, if given, the input
            would be padded with zero or truncated to length n before performing FFT.
            Should be a integer greater than 1.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The transformed input.

        Examples
        --------
        >>> a = ivy.array(np.array([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]))
        >>> b = ivy.array(np.exp(2j * np.pi * np.arange(8) / 8))
        >>> c = ivy.Container(a=a, b=b)
        >>> dims = ivy.Container(a=0, b=0)
        >>> ivy.Container.static_fft(c, dims)
        {
        a: ivy.array([0.+0.j, 12.+0.j, 8.+0.j, 4.+0.j]),
        b: ivy.array([-3.44509285e-16+1.14423775e-17j, 8.00000000e+00-8.11483250e-16j,
                       2.33486982e-16+1.22464680e-16j, 0.00000000e+00+1.22464680e-16j,
                       9.95799250e-17+2.33486982e-16j, 0.00000000e+00+7.66951701e-17j,
                       1.14423775e-17+1.22464680e-16j, 0.00000000e+00+1.22464680e-16j])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fft",
            x,
            dim,
            norm=norm,
            n=n,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def fft(
        self: ivy.Container,
        dim: int,
        /,
        *,
        norm: str = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Array] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.fft. This method simply wraps the
        function, and so the docstring for ivy.fft also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Container containing input volumes *[...,d_in,...]*,
            where d_in indicates the dimension that needs FFT.
        dim
            The dimension along which to take the one dimensional FFT.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        n
            Optional argument indicating the sequence length, if given, the input would
            be padded with zero or truncated to length n before performing FFT.
            Should be a integer greater than 1.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Container containing the transformed inputs.

        Examples
        --------
        >>> a = ivy.array(np.array([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]))
        >>> b = ivy.array(np.exp(2j * np.pi * np.arange(8) / 8))
        >>> c = ivy.Container(a=a, b=b)
        >>> dims = ivy.Container(a=0, b=0)
        >>> c.fft(dims)
        {
        a: ivy.array([0.+0.j, 12.+0.j, 8.+0.j, 4.+0.j]),
        b: ivy.array([-3.44509285e-16+1.14423775e-17j, 8.00000000e+00-8.11483250e-16j,
                       2.33486982e-16+1.22464680e-16j, 0.00000000e+00+1.22464680e-16j,
                       9.95799250e-17+2.33486982e-16j, 0.00000000e+00+7.66951701e-17j,
                       1.14423775e-17+1.22464680e-16j, 0.00000000e+00+1.22464680e-16j])
        }
        """
        return self._static_fft(
            self,
            dim,
            norm=norm,
            n=n,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_ifft(
        x: ivy.Container,
        dim: int,
        *,
        norm: str = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container static method variant of ivy.ifft. This method simply wraps the
        function, and so the docstring for ivy.ifft also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Container containing input volumes *[...,d_in,...]*,
            where d_in indicates the dimension that needs IFFT.
        dim
            The dimension along which to take the one dimensional IFFT.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        n
            Optional argument indicating the sequence length, if given, the input would
            be padded with zero or truncated to length n before performing IFFT.
            Should be a integer greater than 1.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The transformed input.

        Examples
        --------
        >>> a = ivy.array(np.array([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]))
        >>> b = ivy.array(np.exp(2j * np.pi * np.arange(8) / 8))
        >>> c = ivy.Container(a=a, b=b)
        >>> dims = ivy.Container(a=0, b=0)
        >>> ivy.Container.static_ifft(c, dims)
        {
        a: ivy.array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j]),
        b: ivy.array([-4.30636606e-17+1.43029718e-18j, 0.00000000e+00+1.53080850e-17j,
                       1.43029718e-18+1.53080850e-17j, 0.00000000e+00+9.58689626e-18j,
                       1.24474906e-17+2.91858728e-17j, 0.00000000e+00+1.53080850e-17j,
                       2.91858728e-17+1.53080850e-17j, 1.00000000e+00-1.01435406e-16j])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "ifft",
            x,
            dim,
            norm=norm,
            n=n,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ifft(
        self: ivy.Container,
        dim: int,
        *,
        norm: str = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Array] = None,
    ):
        """
        ivy.Container instance method variant of ivy.ifft. This method simply wraps the
        function, and so the docstring for ivy.ifft also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Container containing input volumes *[...,d_in,...]*,
            where d_in indicates the dimension that needs IFFT.
        dim
            The dimension along which to take the one dimensional IFFT.
        norm
            Optional argument, "backward", "ortho" or "forward". Defaults to be
            "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        n
            Optional argument indicating the sequence length, if given, the input
            would be padded with zero or truncated to length n before performing IFFT.
            Should be a integer greater than 1.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Container containing the transformed inputs.

        Examples
        --------
        >>> a = ivy.array(np.array([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]))
        >>> b = ivy.array(np.exp(2j * np.pi * np.arange(8) / 8))
        >>> c = ivy.Container(a=a, b=b)
        >>> dims = ivy.Container(a=0, b=0)
        >>> c.ifft(dims)
        {
        a: ivy.array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j]),
        b: ivy.array([-4.30636606e-17+1.43029718e-18j, 0.00000000e+00+1.53080850e-17j,
                       1.43029718e-18+1.53080850e-17j, 0.00000000e+00+9.58689626e-18j,
                       1.24474906e-17+2.91858728e-17j, 0.00000000e+00+1.53080850e-17j,
                       2.91858728e-17+1.53080850e-17j, 1.00000000e+00-1.01435406e-16j])
        }
        """
        return self.static_ifft(
            self,
            dim,
            norm=norm,
            n=n,
            out=out,
        )

    @staticmethod
    def static_embedding(
        weight: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        indices: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        max_norm: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "embedding",
            weight,
            indices,
            max_norm=max_norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def embedding(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        indices: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        max_norm: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_embedding(
            self,
            indices,
            max_norm=max_norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_dft(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = 1,
        inverse: bool = False,
        onesided: bool = False,
        dft_length: Optional[Union[int, Tuple[int]]] = None,
        norm: str = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """

        Parameters
        ----------
        x
        axis
        inverse
        onesided
        dft_length
        norm
        key_chains
        to_apply
        prune_unapplied
        map_sequences
        out

        """
        return ContainerBase.cont_multi_map_in_function(
            "dft",
            x,
            axis=axis,
            inverse=inverse,
            onesided=onesided,
            dft_length=dft_length,
            norm=norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def dft(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = 1,
        inverse: bool = False,
        onesided: bool = False,
        dft_length: Optional[Union[int, Tuple[int]]] = None,
        norm: str = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """

        Parameters
        ----------
        axis
        inverse
        onesided
        dft_length
        norm
        key_chains
        to_apply
        prune_unapplied
        map_sequences
        out

        """
        return self.static_dft(
            self,
            axis=axis,
            inverse=inverse,
            onesided=onesided,
            dft_length=dft_length,
            norm=norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_interpolate(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        size: Union[Sequence[int], int],
        /,
        *,
        mode: Literal[
            "linear",
            "bilinear",
            "trilinear",
            "nearest",
            "area",
            "nearest_exact",
            "tf_area",
            "bicubic",
        ] = "linear",
        scale_factor: Optional[Union[Sequence[int], int]] = None,
        recompute_scale_factor: Optional[bool] = None,
        align_corners: Optional[bool] = None,
        antialias: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Down/up samples the input to the given size. The algorithm used for
        interpolation is determined by mode.

        Parameters
        ----------
        x
            Input array, Must have the shape
            [batch x channels x [optional depth] x [optional height] x width].
        size
            Output size.
        mode
            Interpolation mode. Can be one of the following:
            - linear
            - bilinear
            - trilinear
            - nearest
            - area
            - tf_area
            - bicubic
            - mitchellcubic
            - lanczos3
            - lanczos5
            - gaussian
        scale_factor
            Multiplier for spatial size that defines the output
            size (overwriting `size`).
        align_corners
            If True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at the corner pixels. If False, the corner
            pixels are not aligned, and the interpolation uses edge value padding for
            out-of-boundary values.
            only has an effect when mode is 'linear', 'bilinear',
            'bicubic' or 'trilinear'. Default: False
        antialias
            If True, antialiasing is applied when downsampling an image.
            Supported modes: 'bilinear', 'bicubic'.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
            resized array
        """
        return ContainerBase.cont_multi_map_in_function(
            "interpolate",
            x,
            size,
            mode=mode,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute_scale_factor,
            align_corners=align_corners,
            antialias=antialias,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def interpolate(
        self: ivy.Container,
        size: Union[Sequence[int], int],
        /,
        *,
        mode: Literal[
            "linear",
            "bilinear",
            "trilinear",
            "nearest",
            "area",
            "nearest_exact",
            "tf_area",
            "bicubic",
        ] = "linear",
        scale_factor: Optional[Union[Sequence[int], int]] = None,
        recompute_scale_factor: Optional[bool] = None,
        align_corners: Optional[bool] = None,
        antialias: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Down/up samples the input to the given size. The algorithm used for
        interpolation is determined by mode.

        Parameters
        ----------
        x
            Input array, Must have the shape
            [batch x channels x [optional depth] x [optional height] x width].
        size
            Output size.
        mode
            Interpolation mode. Can be one of the following:
            - linear
            - bilinear
            - trilinear
            - nearest
            - area
            - tf_area
            - bicubic
            - mitchellcubic
            - lanczos3
            - lanczos5
            - gaussian
        scale_factor
            Multiplier for spatial size that defines the output
            size (overwriting `size`).
        align_corners
            If True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at the corner pixels. If False, the corner
            pixels are not aligned, and the interpolation uses edge value padding for
            out-of-boundary values.
            only has an effect when mode is 'linear', 'bilinear',
            'bicubic' or 'trilinear'. Default: False
        antialias
            If True, antialiasing is applied when downsampling an image.
            Supported modes: 'bilinear', 'bicubic'.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
            resized array
        """
        return self.static_interpolate(
            self,
            size,
            mode=mode,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute_scale_factor,
            align_corners=align_corners,
            antialias=antialias,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_adaptive_avg_pool1d(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        output_size: int,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.adaptive_avg_pool1d. This method
        simply wraps the function, and so the docstring for ivy.adaptive_avg_pool1d also
        applies to this method with minimal changes.

        Parameters
        ----------
        input
            Input array. Must have shape (N, C, L_in) or (C, L_in) where N is
            the batch dimension, C is the feature dimension, and L_in is the spatial
            dimension.
        output_size
            Spatial output size.

        Returns
        -------
            The result of the pooling operation. Will have shape (N, C, L_out) or
            (C, L_out), where L_out = `output_size`
        """
        return ContainerBase.cont_multi_map_in_function(
            "adaptive_avg_pool1d",
            input,
            output_size,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adaptive_avg_pool1d(
        self: ivy.Container,
        output_size: int,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        Apply a 1D adaptive average pooling over an input signal composed of several
        input planes.

        Parameters
        ----------
        self
            Input container.
        output_size
            Spatial output size.

        Returns
        -------
            The result of the pooling operation.
        """
        return self.static_adaptive_avg_pool1d(
            self,
            output_size,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_adaptive_avg_pool2d(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        output_size: Union[Sequence[int], int],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.adaptive_avg_pool2d. This method
        simply wraps the function, and so the docstring for ivy.adaptive_avg_pool2d also
        applies to this method with minimal changes.

        Parameters
        ----------
        input
            Input array. Must have shape (N, C, H_in, W_in) or (C, H_in, W_in) where N
            is the batch dimension, C is the feature dimension, and H_in and W_in are
            the 2 spatial dimensions.
        output_size
            Spatial output size.

        Returns
        -------
            The result of the pooling operation. Will have shape (N, C, S_0, S_1) or
            (C, S_0, S_1), where S = `output_size`
        """
        return ContainerBase.cont_multi_map_in_function(
            "adaptive_avg_pool2d",
            input,
            output_size,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adaptive_avg_pool2d(
        self: ivy.Container,
        output_size: int,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        Apply a 2D adaptive average pooling over an input signal composed of several
        input planes.

        Parameters
        ----------
        self
            Input container.
        output_size
            Spatial output size.

        Returns
        -------
            The result of the pooling operation.
        """
        return self.static_adaptive_avg_pool2d(
            self,
            output_size,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_ifftn(
        x: ivy.Container,
        s: Optional[Union[int, Tuple[int, ...]]] = None,
        axes: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        norm: str = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container static method variant of ivy.ifftn.

        This method simply wraps the function, and so the docstring for
        ivy.ifftn  also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of complex numbers.

        s
            sequence of ints, optional
            Shape (length of transformed axis) of the output (`s[0]` refers to axis 0,
            `s[1]` to axis 1, etc.). If given shape is smaller than that of the input,
            the input is cropped. If larger, input is padded with zeros. If `s` is not
            given, shape of input along axes specified by axes is used.
        axes
            axes over which to compute the IFFT. If not given, last `len(s)` axes are
            used, or all axes if `s` is also not specified. Repeated indices in axes
            means inverse transform over that axis is performed multiple times.
        norm
            Optional argument, "backward", "ortho" or "forward".
            Defaults to be "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The truncated or zero-padded input, transformed along the axes indicated
            by axes, or by a combination of s or x, as explained in the parameters
            section above.
        """
        return ContainerBase.cont_multi_map_in_function(
            "ifftn",
            x,
            s=s,
            axes=axes,
            norm=norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ifftn(
        self: ivy.Container,
        s: Optional[Union[int, Tuple[int, ...]]] = None,
        axes: Optional[Union[int, Tuple[int, ...]]] = None,
        *,
        norm: str = "backward",
        out: Optional[ivy.Array] = None,
    ):
        """
        ivy.Container static method variant of ivy.ifftn.

        This method simply wraps the function, and so the docstring for
        ivy.ifftn also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of complex numbers.

        s
            sequence of ints, optional
            Shape (length of transformed axis) of the output (`s[0]` refers to axis 0,
            `s[1]` to axis 1, etc.). If given shape is smaller than that of the input,
            the input is cropped. If larger, input is padded with zeros. If `s` is not
            given, shape of input along axes specified by axes is used.
        axes
            axes over which to compute the IFFT. If not given, last `len(s)` axes are
            used, or all axes if `s` is also not specified. Repeated indices in axes
            means inverse transform over that axis is performed multiple times.
        norm
            Optional argument, "backward", "ortho" or "forward".
            Defaults to be "backward".
            "backward" indicates no normalization.
            "ortho" indicates normalization by 1/sqrt(n).
            "forward" indicates normalization by 1/n.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Container containing the transformed inputs
        """
        return self.static_ifftn(
            self,
            s=s,
            axes=axes,
            norm=norm,
            out=out,
        )
