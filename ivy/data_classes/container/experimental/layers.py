# global
from typing import Optional, Union, List, Dict, Tuple, Literal, Sequence

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithLayersExperimental(ContainerBase):
    @staticmethod
    def static_max_pool1d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int, ...], ivy.Container],
        strides: Union[int, Tuple[int, ...], ivy.Container],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]], ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NWC",
        dilation: Union[int, Tuple[int], ivy.Container] = 1,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
            "SAME" or "VALID" indicating the algorithm, or list
            indicating the per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
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
        kernel: Union[int, Tuple[int, ...], ivy.Container],
        strides: Union[int, Tuple[int, ...], ivy.Container],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]], ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NWC",
        dilation: Union[int, Tuple[int], ivy.Container] = 1,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
            "NWC" or "NCW". Defaults to "NWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
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
        kernel: Union[int, Tuple[int, ...], ivy.Container],
        strides: Union[int, Tuple[int, ...], ivy.Container],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]], ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NHWC",
        dilation: Union[int, Tuple[int, ...], ivy.Container] = 1,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
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
        kernel: Union[int, Tuple[int, ...], ivy.Container],
        strides: Union[int, Tuple[int, ...], ivy.Container],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]], ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NHWC",
        dilation: Union[int, Tuple[int, ...], ivy.Container] = 1,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
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
        kernel: Union[int, Tuple[int, ...], ivy.Container],
        strides: Union[int, Tuple[int, ...], ivy.Container],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]], ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NDHWC",
        dilation: Union[int, Tuple[int, ...], ivy.Container] = 1,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
            "NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
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
            dilation=dilation,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def max_pool3d(
        self: ivy.Container,
        kernel: Union[int, Tuple[int, ...], ivy.Container],
        strides: Union[int, Tuple[int, ...], ivy.Container],
        padding: Union[str, int, Tuple[int], List[Tuple[int, int]], ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NDHWC",
        dilation: Union[int, Tuple[int, ...], ivy.Container] = 1,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
            "NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilaton
            The stride between elements within a sliding window, must be > 0.
        ceil_mode
            If True, ceil is used instead of floor to compute the output shape.
            This ensures that every element is covered by a sliding window.
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
            dilation=dilation,
            ceil_mode=ceil_mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_avg_pool1d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[int, Tuple[int], ivy.Container],
        strides: Union[int, Tuple[int], ivy.Container],
        padding: Union[str, ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NWC",
        count_include_pad: Union[bool, ivy.Container] = False,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        kernel: Union[int, Tuple[int], ivy.Container],
        strides: Union[int, Tuple[int], ivy.Container],
        padding: Union[str, ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NWC",
        count_include_pad: Union[bool, ivy.Container] = False,
        ceil_mode: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        kernel: Union[int, Tuple[int], Tuple[int, int], ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int], ivy.Container],
        padding: Union[str, ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NHWC",
        count_include_pad: Union[bool, ivy.Container] = False,
        ceil_mode: Union[bool, ivy.Container] = False,
        divisor_override: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        kernel: Union[int, Tuple[int], Tuple[int, int], ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int], ivy.Container],
        padding: Union[str, ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NHWC",
        count_include_pad: Union[bool, ivy.Container] = False,
        ceil_mode: Union[bool, ivy.Container] = False,
        divisor_override: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        kernel: Union[int, Tuple[int], Tuple[int, int, int], ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int, int], ivy.Container],
        padding: Union[str, ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NDHWC",
        count_include_pad: Union[bool, ivy.Container] = False,
        ceil_mode: Union[bool, ivy.Container] = False,
        divisor_override: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        kernel: Union[int, Tuple[int], Tuple[int, int, int], ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int, int], ivy.Container],
        padding: Union[str, ivy.Container],
        /,
        *,
        data_format: Union[str, ivy.Container] = "NDHWC",
        count_include_pad: Union[bool, ivy.Container] = False,
        ceil_mode: Union[bool, ivy.Container] = False,
        divisor_override: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        type: Union[Literal[1, 2, 3, 4], ivy.Container] = 2,
        n: Optional[Union[int, ivy.Container]] = None,
        axis: Union[int, ivy.Container] = -1,
        norm: Optional[Union[Literal["ortho"], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        type: Union[Literal[1, 2, 3, 4], ivy.Container] = 2,
        n: Optional[Union[int, ivy.Container]] = None,
        axis: Union[int, ivy.Container] = -1,
        norm: Optional[Union[Literal["ortho"], ivy.Container]] = None,
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
        type: Union[Literal[1, 2, 3, 4], ivy.Container] = 2,
        n: Optional[Union[int, ivy.Container]] = None,
        axis: Union[int, ivy.Container] = -1,
        norm: Optional[Union[Literal["ortho"], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        type: Union[Literal[1, 2, 3, 4], ivy.Container] = 2,
        n: Optional[Union[int, ivy.Container]] = None,
        axis: Union[int, ivy.Container] = -1,
        norm: Optional[Union[Literal["ortho"], ivy.Container]] = None,
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
        dim: Union[int, ivy.Container],
        /,
        *,
        norm: Union[str, ivy.Container] = "backward",
        n: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        dim: Union[int, ivy.Container],
        /,
        *,
        norm: Union[str, ivy.Container] = "backward",
        n: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        dim: Union[int, ivy.Container],
        *,
        norm: Union[str, ivy.Container] = "backward",
        n: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        dim: Union[int, ivy.Container],
        *,
        norm: Union[str, ivy.Container] = "backward",
        n: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
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
        max_norm: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        max_norm: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        axis: Union[int, ivy.Container] = 1,
        inverse: Union[bool, ivy.Container] = False,
        onesided: Union[bool, ivy.Container] = False,
        dft_length: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        norm: Union[str, ivy.Container] = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
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
        axis: Union[int, ivy.Container] = 1,
        inverse: Union[bool, ivy.Container] = False,
        onesided: Union[bool, ivy.Container] = False,
        dft_length: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        norm: Union[str, ivy.Container] = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
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
        size: Union[Sequence[int], int, ivy.Container],
        /,
        *,
        mode: Union[
            Literal[
                "linear",
                "bilinear",
                "trilinear",
                "nearest",
                "area",
                "nearest_exact",
                "tf_area",
                "bicubic",
            ],
            ivy.Container,
        ] = "linear",
        scale_factor: Optional[Union[Sequence[int], int, ivy.Container]] = None,
        recompute_scale_factor: Optional[Union[bool, ivy.Container]] = None,
        align_corners: Optional[Union[bool, ivy.Container]] = None,
        antialias: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        size: Union[Sequence[int], int, ivy.Container],
        /,
        *,
        mode: Union[
            Literal[
                "linear",
                "bilinear",
                "trilinear",
                "nearest",
                "area",
                "nearest_exact",
                "tf_area",
                "bicubic",
            ],
            ivy.Container,
        ] = "linear",
        scale_factor: Optional[Union[Sequence[int], int, ivy.Container]] = None,
        recompute_scale_factor: Optional[Union[bool, ivy.Container]] = None,
        align_corners: Optional[Union[bool, ivy.Container]] = None,
        antialias: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        output_size: Union[int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        output_size: Union[int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        output_size: Union[Sequence[int], int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        output_size: Union[int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
    def static_adaptive_max_pool2d(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        output_size: Union[Sequence[int], int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.adaptive_max_pool2d. This method
        simply wraps the function, and so the docstring for ivy.adaptive_max_pool2d also
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
            "adaptive_max_pool2d",
            input,
            output_size,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adaptive_max_pool2d(
        self: ivy.Container,
        output_size: Union[int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """
        Apply a 2D adaptive maximum pooling over an input signal composed of several
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
        return self.static_adaptive_max_pool2d(
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
        s: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        axes: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        *,
        norm: Union[str, ivy.Container] = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
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
        s: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        axes: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        *,
        norm: Union[str, ivy.Container] = "backward",
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
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


    @staticmethod
    def stft(
        signal: Union[ivy.Array, ivy.NativeArray],
        n_fft: Optional[Union[int, Tuple[int]]],
        frame_step: int,
        /,
        *,
        axis: Optional[int] = None,
        onesided:Optional[bool] = True,
        fs: Optional[float] = 1.0,
        window: Optional[Union[ivy.Array, list, str, Tuple[int]]] = None,
        win_length: Optional[int] = None,
        noverlap: Optional[int] = None,
        center: Optional[bool] = False,
        pad_mode: Optional[str] = "reflect",
        normalized: Optional[bool] = False,
        detrend: Optional[Union[str, callable, bool]] = False,
        return_complex: Optional[bool] = True,
        boundary: Optional[str] = 'zeros',
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Compute the Short-time Fourier transform  of input.
        
        Parameters
        ----------
        signal
            Input tensor representing a real or complex valued signal. 
            For real input, the following shape is expected: [batch_
            size][signal_length][1]. For complex input, the following 
            shape is expected: [batch_size][signal_length][2], where 
            [batch_size][signal_length][0] represents the real component 
            and [batch_size][signal_length][1] represents the imaginary 
            component of the signal.        
        n_fft
           Size of Fourier transform.
        frame_step
           An integer scalar Tensor. The number of samples to step.             
        axis
            The axis on which to perform the DFT. By default this
            value is  set to 1, which corresponds to the first dimension
            after the batch index.
        onesided
            If onesided is True, only values for w in [0, 1, 2, …, floor
            (n_fft/2) + 1] are returned because the real-to-complex Fourier 
            transform satisfies the conjugate symmetry, i.e., X[m, w] = 
            X[m,w]=X[m,n_fft-w]*. Note if the input or window tensors are 
            complex, then onesided output is not possible.Enabling onesided 
            with real inputs performs a Real-valued fast Fourier transform 
            (RFFT). When invoked with real or complex valued input, the
            default value is False. Values can be True or False.
        fs
            Sampling frequency of the x time series. Defaults to 1.0.
        window
            Desired window to use. If window is a string or tuple, 
            it is passed to get_window to generate the window values, 
            which are DFT-even by default. See get_window for a list of 
            windows and required parameters. If window is array_like it 
            will be used directly as the window and its length must be 
            nperseg. Defaults to a Hann window.
        win_length
            The size of window frame and STFT filter. Defaults to None.    
        noverlap
            Number of points to overlap between segments. If None, 
            noverlap = nperseg // 2. Defaults to None.
        center  
            Whether to pad x to make that the t * hop_length at the 
            center of t-th frame. Default: True.          
        pad_mode 
            Choose padding pattern when center is True. See paddle.
            nn.functional.pad for all padding options. Default: “reflect”.
        normalized 
            Control whether to scale the output by 1/sqrt(n_fft). 
            Default: False
        detrend 
            Specifies how to detrend each segment. If detrend is a string, 
            it is passed as the type argument to the detrend function. If 
            it is a function, it takes a segment and returns a detrended 
            segment. If detrend is False, no detrending is done. Defaults 
            to False.
        return_complex
            Whether to return a complex tensor, or a real tensor with an extra 
            last dimension for the real and imaginary components.            
        boundary
            Specifies whether the input signal is extended at both ends, and 
            how to generate the new values, in order to center the first 
            windowed segment on the first input point. This has the benefit of 
            enabling reconstruction of the first input point when the employed 
            window function starts at zero. Valid options are ['even', 'odd', 
            'constant','zeros', None]. Defaults to ‘zeros’, for zero padding 
            extension. I.e. [1, 2, 3, 4] is extended to [0, 1, 2, 3, 4, 0] 
            for nperseg=3.     
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.
        Returns
        -------
        ret
            The Short-time Fourier Transform of the signals.If onesided is 1, the 
            output has the shape: [batch_size][frames][dft_unique_bins][2], where 
            dft_unique_bins is frame_length // 2 + 1 (the unique components of 
            the DFT) If onesided is 0, the output has the shape: [batch_size]
            [frames][frame_length][2], where frame_length is the length of 
            the DFT.
        """
        return self.static_stft(
            signal,
            n_fft,
            frame_step,
            axis=axis,
            onesided=onesided,
            fs=fs,
            window=window,
            win_length=win_length,
            noverlap=noverlap,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            detrend=detrend,
            return_complex=return_complex,
            boundary=boundary,
            out=out,
        )

    @staticmethod
    def static_rfftn(
        x: ivy.Container,
        s: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        axes: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        *,
        norm: Union[str, ivy.Container] = "backward",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.rfftn.

        This method simply wraps the function, and so the docstring for
        ivy.rfftn also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of real numbers.

        s
            sequence of ints, optional
            Shape (length of transformed axis) to use from the input (`s[0]` refers to
            axis 0,`s[1]` to axis 1, etc.). The final element of `s` corresponds to `n`
            for `rfft(x, n)`, while for the remaining axes, it corresponds to `n` for
            `fft(x, n)`. Along any axis, if the given shape is smaller than that of the
            input, the input is cropped. If it is larger,the input is padded with zeros.
            If `s` is not given, the shape of the input along the axes specified by
            `axes` is used.

        axes
            sequence of ints, optional
            Axes over which to compute the FFT. If not given, the last `len(s)` axes
            are used, or all axes if `s` is also not specified.

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
            The truncated or zero-padded input, transformed along the axes indicated by
            `axes` or by a combination of `s` or `x`, as explained in the parameters
            section above.
        """
        return ContainerBase.cont_multi_map_in_function(
            "rfftn",
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

    def rfftn(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        s: Optional[Union[Sequence[int], ivy.Container]] = None,
        axes: Optional[Union[int, Tuple[int], ivy.Container]] = None,
        *,
        norm: Union[str, ivy.Container] = "backward",
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        Compute the n-dimensional discrete Fourier Transform for real input.

        Parameters
        ----------
        axes : int or tuple of ints, optional
            Axes over which to compute the FFT. If not given, the last `n` axes are
            used.
        s : sequence of ints, optional
            Shape (length of each transformed axis) of the output. Along each axis,
            if the given shape is smaller than
            that of the input, the input is cropped. If it is larger, the input
            is padded with zeros.
        norm : {'backward', 'ortho', 'forward'}, optional
            Normalization mode. Default is 'backward'.
        out : array-like, optional
            Output array. Must have the same shape and type as the expected output.

        Returns
        -------
        transformed : Container
            The n-dimensional discrete Fourier Transform of the input.
        """
        return self.static_rfftn(
            self,
            s=s,
            axes=axes,
            norm=norm,
            out=out,
        )
