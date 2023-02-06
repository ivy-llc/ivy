# global
from typing import Optional, Union, List, Dict, Tuple, Literal

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithLayersExperimental(ContainerBase):
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
        return ContainerBase.cont_multi_map_in_function(
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
        return ContainerBase.cont_multi_map_in_function(
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
        """ivy.Container static method variant of ivy.max_pool3d. This method simply
        wraps the function, and so the docstring for ivy.max_pool3d also applies
        to this method with minimal changes.

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
        """ivy.Container static method variant of ivy.max_pool3d. This method simply
        wraps the function, and so the docstring for ivy.max_pool3d also applies
        to this method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.avg_pool1d. This method simply
        wraps the function, and so the docstring for ivy.avg_pool1d also applies
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of `ivy.avg_pool1d`. This method simply
        wraps the function, and so the docstring for `ivy.avg_pool1d` also applies
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.avg_pool2d. This method simply
        wraps the function, and so the docstring for ivy.avg_pool2d also applies
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of `ivy.avg_pool2d`. This method simply
        wraps the function, and so the docstring for `ivy.avg_pool2d` also applies
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.avg_pool3d. This method simply
        wraps the function, and so the docstring for ivy.avg_pool3d also applies
        to this method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.avg_pool3d. This method simply
        wraps the function, and so the docstring for ivy.avg_pool3d also applies
        to this method with minimal changes.

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
        return self.static_avg_pool3d(
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
    def static_dct(
        x: ivy.Container,
        /,
        *,
        type: Optional[Literal[1, 2, 3, 4]] = 2,
        n: Optional[int] = None,
        axis: Optional[int] = -1,
        norm: Optional[Literal["ortho"]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.dct. This method simply wraps
        the function, and so the docstring for ivy.dct also applies to this method
        with minimal changes.

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
        type: Optional[Literal[1, 2, 3, 4]] = 2,
        n: Optional[int] = None,
        axis: Optional[int] = -1,
        norm: Optional[Literal["ortho"]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.dct. This method simply wraps
        the function, and so the docstring for ivy.dct also applies to this method
        with minimal changes.

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
    def static_fft(
        x: ivy.Container,
        dim: int,
        /,
        *,
        norm: Optional[str] = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ):
        """ivy.Container static method variant of ivy.fft. This method simply wraps
        the function, and so the docstring for ivy.fft also applies to this method
        with minimal changes.

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
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fft(
        self: ivy.Container,
        dim: int,
        /,
        *,
        norm: Optional[str] = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Array] = None,
    ):
        """ivy.Container instance method variant of ivy.fft. This method simply wraps
        the function, and so the docstring for ivy.fft also applies to this method
        with minimal changes.

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
        return self.static_fft(
            self,
            dim,
            norm=norm,
            n=n,
            out=out,
        )

    @staticmethod
    def static_ifft(
        x: ivy.Container,
        dim: int,
        *,
        norm: Optional[str] = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ):
        """ivy.Container static method variant of ivy.ifft. This method simply wraps
        the function, and so the docstring for ivy.ifft also applies to this method
        with minimal changes.

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
        norm: Optional[str] = "backward",
        n: Optional[Union[int, Tuple[int]]] = None,
        out: Optional[ivy.Array] = None,
    ):
        """ivy.Container instance method variant of ivy.ifft. This method simply wraps
        the function, and so the docstring for ivy.ifft also applies to this method
        with minimal changes.

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
        norm: Optional[str] = "backward",
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
        norm: Optional[str] = "backward",
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
