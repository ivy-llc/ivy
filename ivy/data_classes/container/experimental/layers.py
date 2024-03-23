# global
from typing import Optional, Union, List, Dict, Tuple, Literal, Sequence, Callable

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
        """ivy.Container static method variant of ivy.max_pool1d. This method
        simply wraps the function, and so the docstring for ivy.max_pool1d also
        applies to this method with minimal changes.

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
        """ivy.Container instance method variant of `ivy.max_pool1d`. This
        method simply wraps the function, and so the docstring for
        `ivy.max_pool1d` also applies to this method with minimal changes.

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
        """ivy.Container static method variant of ivy.max_pool2dd. This method
        simply wraps the function, and so the docstring for ivy.max_pool2d also
        applies to this method with minimal changes.

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
        """ivy.Container instance method variant of `ivy.max_pool2d`. This
        method simply wraps the function, and so the docstring for
        `ivy.max_pool2d` also applies to this method with minimal changes.

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
        >>> a = ivy.arange(24.).reshape((1, 2, 3, 4))
        >>> b = ivy.arange(48.).reshape((2, 4, 3, 2))
        >>> x = ivy.Container(a=a, b=b)
        >>> y = x.max_pool2d(3, 1, "VALID")
        >>> print(y)
        {
            a: ivy.array([], shape=(1, 0, 1, 4)),
            b: ivy.array([[[[16., 17.]],
                           [[22., 23.]]],
                         [[[40., 41.]],
                           [[46., 47.]]]])
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
        """ivy.Container static method variant of ivy.max_pool3d. This method
        simply wraps the function, and so the docstring for ivy.max_pool3d also
        applies to this method with minimal changes.

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
        """ivy.Container static method variant of ivy.max_pool3d. This method
        simply wraps the function, and so the docstring for ivy.max_pool3d also
        applies to this method with minimal changes.

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
        >>> a = ivy.arange(24.).reshape((1, 2, 3, 4, 1))
        >>> b = ivy.arange(48.).reshape((2, 4, 3, 2, 1))
        >>> x = ivy.Container(a=a, b=b)
        >>> print(x.max_pool3d(3, 1, "VALID"))
        {
            a: ivy.array([], shape=(1, 0, 1, 2, 1)),
            b: ivy.array([], shape=(2, 2, 1, 0, 1))
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
        """ivy.Container static method variant of ivy.avg_pool1d. This method
        simply wraps the function, and so the docstring for ivy.avg_pool1d also
        applies to this method with minimal changes.

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
        """ivy.Container instance method variant of `ivy.avg_pool1d`. This
        method simply wraps the function, and so the docstring for
        `ivy.avg_pool1d` also applies to this method with minimal changes.

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
        """ivy.Container static method variant of ivy.avg_pool2d. This method
        simply wraps the function, and so the docstring for ivy.avg_pool2d also
        applies to this method with minimal changes.

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
        >>> y = ivy.Container.static_avg_pool2d(x, (2, 2), (1, 1), "SAME")
        >>> print(y)
        {
            a: (<class ivy.data_classes.array.array.Array> shape=[2, 1, 3, 2]),
            b: (<class ivy.data_classes.array.array.Array> shape=[2, 4, 3, 2])
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
        """ivy.Container instance method variant of `ivy.avg_pool2d`. This
        method simply wraps the function, and so the docstring for
        `ivy.avg_pool2d` also applies to this method with minimal changes.

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
        >>> y = x.avg_pool2d(2, 1, "SAME")
        >>> print(y)
        {
            a: (<class ivy.data_classes.array.array.Array> shape=[2, 1, 3, 2]),
            b: (<class ivy.data_classes.array.array.Array> shape=[2, 4, 3, 2])
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
        """ivy.Container static method variant of ivy.avg_pool3d. This method
        simply wraps the function, and so the docstring for ivy.avg_pool3d also
        applies to this method with minimal changes.

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
        """ivy.Container static method variant of ivy.avg_pool3d. This method
        simply wraps the function, and so the docstring for ivy.avg_pool3d also
        applies to this method with minimal changes.

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
        >>> a = ivy.arange(24.).reshape((1, 2, 3, 4, 1))
        >>> b = ivy.arange(48.).reshape((2, 4, 3, 2, 1))
        >>> x = ivy.Container(a=a, b=b)
        >>> print(x.avg_pool3d(2, 1, "VALID"))
        {
            a: ivy.array([[[[[8.5],
                             [9.5],
                             [10.5]],
                            [[12.5],
                             [13.5],
                             [14.5]]]]]),
            b: (<class ivy.data_classes.array.array.Array> shape=[2, 3, 2, 1, 1])
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
        """ivy.Container static method variant of ivy.dct. This method simply
        wraps the function, and so the docstring for ivy.dct also applies to
        this method with minimal changes.

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
        """ivy.Container instance method variant of ivy.dct. This method simply
        wraps the function, and so the docstring for ivy.dct also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Container with the input signals.
        type
            The type of the dct. Must be 1, 2, 3 or 4.
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
        """ivy.Container static method variant of ivy.idct. This method simply
        wraps the function, and so the docstring for ivy.idct also applies to
        this method with minimal changes.

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
        """ivy.Container instance method variant of ivy.idct. This method
        simply wraps the function, and so the docstring for ivy.idct also
        applies to this method with minimal changes.

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
        """ivy.Container static method variant of ivy.fft. This method simply
        wraps the function, and so the docstring for ivy.fft also applies to
        this method with minimal changes.

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
        """ivy.Container instance method variant of ivy.fft. This method simply
        wraps the function, and so the docstring for ivy.fft also applies to
        this method with minimal changes.

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
        """ivy.Container static method variant of ivy.ifft. This method simply
        wraps the function, and so the docstring for ivy.ifft also applies to
        this method with minimal changes.

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
        """ivy.Container instance method variant of ivy.ifft. This method
        simply wraps the function, and so the docstring for ivy.ifft also
        applies to this method with minimal changes.

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
        """Down/up samples the input to the given size. The algorithm used for
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
        """Down/up samples the input to the given size. The algorithm used for
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
        """ivy.Container static method variant of ivy.adaptive_avg_pool1d. This
        method simply wraps the function, and so the docstring for
        ivy.adaptive_avg_pool1d also applies to this method with minimal
        changes.

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
        """Apply a 1D adaptive average pooling over an input signal composed of
        several input planes.

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
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        data_format: str = "NHWC",
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.adaptive_avg_pool2d. This
        method simply wraps the function, and so the docstring for
        ivy.adaptive_avg_pool2d also applies to this method with minimal
        changes.

        Parameters
        ----------
        input
            A 3D or 4D input array. Should have a floating-point data type.
        output_size
            Spatial output size.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".

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
            data_format=data_format,
        )

    def adaptive_avg_pool2d(
        self: ivy.Container,
        output_size: Union[int, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        data_format: str = "NHWC",
    ) -> ivy.Container:
        """Apply a 2D adaptive average pooling over an input signal composed of
        several input planes.

        Parameters
        ----------
        self
            Input container.
        output_size
            Spatial output size.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".

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
            data_format=data_format,
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
        """ivy.Container static method variant of ivy.adaptive_max_pool2d. This
        method simply wraps the function, and so the docstring for
        ivy.adaptive_max_pool2d also applies to this method with minimal
        changes.

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
        """Apply a 2D adaptive maximum pooling over an input signal composed of
        several input planes.

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
    def static_adaptive_max_pool3d(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        output_size: Union[Sequence[int], int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "adaptive_max_pool3d",
            input,
            output_size,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adaptive_max_pool3d(
        self: ivy.Container,
        output_size: Union[int, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        return self.static_adaptive_max_pool3d(
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
        """ivy.Container static method variant of ivy.ifftn.

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
        """ivy.Container static method variant of ivy.ifftn.

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

        Examples
        --------
        >>> x = ivy.Container(
        ...         a=ivy.array([[0.247306+0.908323j, 0.494955+0.90395j,
        ...                       0.98193269+0.49560517j],
        ...                      [0.93280757+0.48075343j, 0.28526384+0.3351205j,
        ...                       0.2343787 +0.83528011j],
        ...                      [0.18791352+0.30690572j, 0.82115787+0.96195183j,
        ...                       0.44719226+0.72654048j]]),
        ...         b=ivy.array([[0.24730653+0.90832391j, 0.49495562+0.9039565j,
        ...                       0.98193269+0.49560517j],
        ...                      [0.93280757+0.48075343j, 0.28526384+0.3351205j,
        ...                       0.2343787 +0.83528011j],
        ...                      [0.18791352+0.30690572j, 0.82115787+0.96195183j,
        ...                       0.44719226+0.72654048j]]),
        ...     )
        >>> y = x.ifftn(s=[2, 1], axes=[0, 1], norm='ortho')
        >>> print(y)
        {
            a: ivy.array([[0.8344667+0.98222595j],
                          [-0.48472244+0.30233797j]]),
            b: ivy.array([[0.8344667+0.98222595j],
                          [-0.48472244+0.30233797j]])
        }
        """
        return self.static_ifftn(
            self,
            s=s,
            axes=axes,
            norm=norm,
            out=out,
        )

    @staticmethod
    def static_rfft(
        x: ivy.Container,
        /,
        *,
        n: Optional[Union[int, ivy.Container]] = None,
        axis: Union[int, ivy.Container] = -1,
        norm: Union[
            Literal["backward", "ortho", "forward"], ivy.Container
        ] = "backward",
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.rfft.

        This method simply wraps the function, and so the docstring for
        ivy.rfft also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array. Must have a real-valued floating-point data type.
        n
            length of the transformed axis of the input. If
            -   n is greater than the length of the input array, the input array
            is zero-padded to length n.
            -   n is less than the length of the input array, the input array is
            trimmed to length n.
            -   n is not provided, the length of the transformed axis of the
            output must equal the length of the input along the axis specified
            by axis. Default is ``None``.
        axis
            axis (dimension) over which to compute the Fourier transform.
            If not set, the last axis (dimension) is used. Default is ``-1``.
        norm
            normalization mode. Should be one of the following modes:
            -   'backward': no normalization.
            -   'ortho': normalize by 1/sqrt(n) (i.e., make the FFT orthonormal).
            -   'forward': normalize by 1/n.
            Default is ``backward``.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            an array transformed along the axis (dimension) indicated by axis.
            The returned array must have a complex-valued floating-point
            data type determined by Type Promotion Rules.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.,1.,2.]),
        ...                   b=ivy.array([3.,4.,5.]))
        >>> y =  ivy.Container.static_rfft(x)
        >>> print(y)
        {
            a: ivy.array([3.+0.j, -1.5+0.8660254j]),
            b: ivy.array([12.+0.j, -1.5+0.8660254j])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "rfft",
            x,
            n=n,
            axis=axis,
            norm=norm,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def rfft(
        self: ivy.Container,
        /,
        *,
        n: Optional[Union[int, ivy.Container]] = None,
        axis: Union[int, ivy.Container] = -1,
        norm: Union[
            Literal["backward", "ortho", "forward"], ivy.Container
        ] = "backward",
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ):
        """ivy.Container instance method variant of ivy.rfft. This method
        simply wraps the function, and so the docstring for ivy.rfft also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Must have a real-valued floating-point data type.
        n
            length of the transformed axis of the input. If
            -   n is greater than the length of the input array, the input array
            is zero-padded to length n.
            -   n is less than the length of the input array, the input array is
            trimmed to length n.
            -   n is not provided, the length of the transformed axis of the
            output must equal the length of the input along the axis specified
            by axis. Default is ``None``.
        axis
            axis (dimension) over which to compute the Fourier transform.
            If not set, the last axis (dimension) is used. Default is ``-1``.
        norm
            normalization mode. Should be one of the following modes:
            -   'backward': no normalization.
            -   'ortho': normalize by 1/sqrt(n) (i.e., make the FFT orthonormal).
            -   'forward': normalize by 1/n.
            Default is ``backward``.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            an array transformed along the axis (dimension) indicated by axis.
            The returned array must have a complex-valued floating-point
            data type determined by Type Promotion Rules.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.,1.,2.]),
        ...                   b=ivy.array([3.,4.,5.]))
        >>> y = x.rfft()
        >>> print(y)
        {
            a: ivy.array([3.+0.j, -1.5+0.8660254j]),
            b: ivy.array([12.+0.j, -1.5+0.8660254j])
        }
        """
        return self.static_rfft(
            self,
            n=n,
            axis=axis,
            norm=norm,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
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
        """ivy.Container static method variant of ivy.rfftn.

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
        """Compute the n-dimensional discrete Fourier Transform for real input.

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

    @staticmethod
    def static_stft(
        signals: ivy.Container,
        frame_length: Union[int, ivy.Container],
        frame_step: Union[int, ivy.Container],
        /,
        *,
        fft_length: Optional[Union[int, ivy.Container]] = None,
        window_fn: Optional[Union[Callable, ivy.Container]] = None,
        pad_end: Optional[Union[bool, ivy.Container]] = False,
        name: Optional[Union[str, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.stft.

        This method simply wraps the function, and so the docstring for
        ivy.stft also applies to this method with minimal changes.

        Parameters
        ----------
        signals
            Input Arrays.
        frame_length
           An integer scalar Tensor. The window length in samples.
        frame_step
            An integer scalar Tensor. The number of samples to step.
        fft_length, optional
            An integer scalar Tensor. The size of the FFT to apply.
            If not provided, uses the smallest power of 2 enclosing frame_length.
        window_fn, optional
            A callable that takes a window length
            and a dtype keyword argument and returns a [window_length]
            Tensor of samples in the provided datatype.
            If set to None, no windowing is used.
        pad_end, optional
            Whether to pad the end of signals with zeros when the provided frame length
            and step produces a frame that lies partially past its end.
        name, optional
            An optional name for the operation.
        out, optional
            Optional output array for writing the result.

        Returns
        -------
        ret
            A [..., frames, fft_unique_bins] Tensor of
            complex64/complex128 STFT values where fft_unique_bins is
            fft_length // 2 + 1 (the unique components of the FFT).
        """
        return ContainerBase.cont_multi_map_in_function(
            "stft",
            signals,
            frame_length,
            frame_step,
            fft_length=fft_length,
            window_fn=window_fn,
            pad_end=pad_end,
            name=name,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def stft(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        frame_length: Union[int, ivy.Container],
        frame_step: Union[int, ivy.Container],
        /,
        *,
        fft_length: Optional[Union[int, ivy.Container]] = None,
        window_fn: Optional[Union[Callable, ivy.Container]] = None,
        pad_end: Optional[Union[bool, ivy.Container]] = False,
        name: Optional[Union[str, ivy.Container]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """Compute the Short-time Fourier Transform of signals.

        Parameters
        ----------
        self
            Input Arrays.
        frame_length
           An integer scalar Tensor. The window length in samples.
        frame_step
            An integer scalar Tensor. The number of samples to step.
        fft_length
            An integer scalar Tensor. The size of the FFT to apply.
            If not provided, uses the smallest power of 2 enclosing frame_length.
        window_fn
            A callable that takes a window length and
            a dtype keyword argument and returns a [window_length] Tensor of
            samples in the provided datatype. If set to None, no windowing is used.
        pad_end
            Whether to pad the end of signals with zeros when the provided frame length
            and step produces a frame that lies partially past its end.
        name
            An optional name for the operation.
        out
            Optional output array for writing the result.

        Returns
        -------
        ret
            A [..., frames, fft_unique_bins] Tensor of
            complex64/complex128 STFT values where fft_unique_bins is
            fft_length // 2 + 1 (the unique components of the FFT).
        """
        return self.static_stft(
            self,
            frame_length,
            frame_step,
            fft_length=fft_length,
            window_fn=window_fn,
            pad_end=pad_end,
            name=name,
            out=out,
        )

    @staticmethod
    def _static_sliding_window(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        window_size: Union[int, Tuple[int, int], Tuple[int, int, int], ivy.Container],
        /,
        *,
        stride: Union[int, Tuple[int, int], ivy.Container] = 1,
        dilation: Union[int, Tuple[int, int], ivy.Container] = 1,
        padding: Union[str, int, Sequence[Tuple[int, int]], ivy.Container] = "VALID",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.sliding_window. This
        method simply wraps the function, and so the docstring for
        ivy.sliding_window also applies to this method with minimal changes.

        Parameters
        ----------
        input
            An array representing the base area on which the window is going to
            slide over.
        window_size
            Size of the sliding window for each dimension of the input.
        stride
            The stride of the sliding window for each dimension of input
        padding
            Either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’
            (no padding), or a sequence of n (low, high) integer pairs that give the
            padding to apply before and after each spatial dimension.
        dilation
            The stride between elements within a sliding window, must be > 0.

        Returns
        -------
        ret
            The result of the sliding window operation.

        Examples
        --------
        >>> x = ivy.Container(
        ...     a=ivy.array([[1, 2, 3, 4],
        ...                  [5, 6, 7, 8],
        ...                  [9, 10, 11, 12]]),
        ...     b=ivy.array([[13, 14, 15, 16],
        ...                  [17, 18, 19, 20],
        ...                  [21, 22, 23, 24]])
        ... )
        >>> result = ivy.Container._static_sliding_window(x, (2, 2))
        >>> print(result)
        {
            a: ivy.array([[[ 1,  2,  5,  6],
                           [ 2,  3,  6,  7],
                           [ 3,  4,  7,  8]],

                           [[ 5,  6,  9, 10],
                           [ 6,  7, 10, 11],
                           [ 7,  8, 11, 12]]]),
            b: ivy.array([[[13, 14, 17, 18],
                            [14, 15, 18, 19],
                            [15, 16, 19, 20]],

                            [[17, 18, 21, 22],
                            [18, 19, 22, 23],
                            [19, 20, 23, 24]]])

        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sliding_window",
            input,
            window_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def sliding_window(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        window_size: Union[int, Tuple[int, int], Tuple[int, int, int], ivy.Container],
        /,
        *,
        stride: Union[int, Tuple[int, int], ivy.Container] = 1,
        dilation: Union[int, Tuple[int, int], ivy.Container] = 1,
        padding: Union[str, int, Sequence[Tuple[int, int]], ivy.Container] = "VALID",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.sliding_window. This
        method simply wraps the function, and so the docstring for
        ivy.sliding_window also applies to this method with minimal changes.

        Parameters
        ----------
        input
            An array representing the base area on which the window is going to
            slide over.
        window_size
            Size of the sliding window for each dimension of the input.
        stride
            The stride of the sliding window for each dimension of input
        padding
            Either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’
            (no padding), or a sequence of n (low, high) integer pairs that give the
            padding to apply before and after each spatial dimension.
        dilation
            The stride between elements within a sliding window, must be > 0.

        Returns
        -------
        ret
            The result of the sliding window operation.

        Examples
        --------
        >>> x = ivy.Container(
        ...     a=ivy.array([[1, 2, 3, 4],
        ...                  [5, 6, 7, 8],
        ...                  [9, 10, 11, 12]]),
        ...     b=ivy.array([[13, 14, 15, 16],
        ...                  [17, 18, 19, 20],
        ...                  [21, 22, 23, 24]])
        ... )
        >>> x.sliding_window((2, 2))
        {
            a: ivy.array([[[ 1,  2,  5,  6],
                           [ 2,  3,  6,  7],
                           [ 3,  4,  7,  8]],
                           [[ 5,  6,  9, 10],
                           [ 6,  7, 10, 11],
                           [ 7,  8, 11, 12]]]),
            b: ivy.array([[[13, 14, 17, 18],
                            [14, 15, 18, 19],
                            [15, 16, 19, 20]],
                            [[17, 18, 21, 22],
                            [18, 19, 22, 23],
                            [19, 20, 23, 24]]])
        }
        """
        return self._static_sliding_window(
            self,
            window_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_max_unpool1d(
        input: ivy.Container,
        indices: ivy.Container,
        kernel_size: Union[Tuple[int], int],
        /,
        *,
        strides: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
        data_format: Union[str, ivy.Container] = "NCW",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.max_unpool1d.

        Parameters
        ----------
        input
            Pooled input image *[batch_size, w, d_in]*.
        indices
            Indices obtained from the corresponding max pooling operation.
        kernel_size
            Size of the kernel i.e., the sliding window for each
            dimension of input. *[w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list
            indicating the per-dimension paddings.
        data_format
            NWC" or "NCW". Defaults to "NCW".
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

        Returns
        -------
        ret
            The result of the unpooling operation.
        """
        return ContainerBase.cont_multi_map_in_function(
            "max_unpool1d",
            input,
            indices,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def max_unpool1d(
        self,
        indices: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel_size: Union[Tuple[int], int],
        /,
        *,
        strides: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
        data_format: Optional[str] = "NCW",
    ) -> ivy.Container:
        """Compute a 1-D max unpooling given the 1-D pooled input x and its
        indices.

        Parameters
        ----------
        self
            Pooled input image *[batch_size, w, d_in]*.
        indices
            Indices obtained from the corresponding max pooling operation.
        kernel_size
            Size of the kernel i.e., the sliding window for each
            dimension of input. *[w]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or list
            indicating the per-dimension paddings.
        data_format
            NWC" or "NCW". Defaults to "NCW".

        Returns
        -------
        ret
            The result of the unpooling operation.
        """
        return self.static_max_unpool1d(
            self,
            indices,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )

    @staticmethod
    def static_rnn(
        step_function: Callable,
        inputs: ivy.Array,
        initial_states: List[ivy.Array],
        /,
        *,
        go_backwards: bool = False,
        mask: Optional[ivy.Array] = None,
        constants: Optional[ivy.Array] = None,
        unroll: bool = False,
        input_length: Optional[int] = None,
        time_major: bool = False,
        zero_output_for_mask: bool = False,
        return_all_outputs: bool = True,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.rnn.

        Parameters
        ----------
        step_function
            RNN step function.
        inputs
            Array of temporal data of shape (samples, time, ...).
        initial_states
            Array with shape (samples, state_size).
        go_backwards
            If True, do the iteration over the time dimension in reverse order and
            return the reversed sequence.
        mask
            Binary array with shape (samples, time, 1), with a zero for every element
            that is masked.
        constants
            List of constant values passed at each step.
        unroll
            Whether to use a pythonic while loop or ivy.while_loop
        input_length
            An integer or 1-D array, depending on whether the time dimension is
            fixed-length. In case of variable length input, it is used for masking in
            case there is no mask specified.
        time_major
            If True, the inputs and outputs will be in shape (timesteps, batch, ...)
            whereas in the False case, it will be (batch, timesteps, ...).
        zero_output_for_mask
            If True, the otput for masked timestep will be zeros, whereas in the False
            case, output from previous timestep is returned
        return_all_outputs
            If True, return the recurrent outputs for all timesteps in the sequence. If
            False, only return the output for the last timestep.

        Returns
        -------
        ret
            A tuple of
            -   the latest output of the rnn of shape (samples, ...)
            -   the output of the rnn of shape (samples, time, ...) if
                return_all_outputs=True else (samples, 1, ...)
            -   list of tensors, latest states returned by the step funciton, of shape
                (samples, ...)
        """
        return ContainerBase.cont_multi_map_in_function(
            "rnn",
            step_function,
            inputs,
            initial_states,
            go_backwards=go_backwards,
            mask=mask,
            constants=constants,
            unroll=unroll,
            input_length=input_length,
            time_major=time_major,
            zero_output_for_mask=zero_output_for_mask,
            return_all_outputs=return_all_outputs,
        )
