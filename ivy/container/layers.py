# global
from typing import Optional, Tuple, Union, List, Callable, Dict

# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithLayers(ContainerBase):
    @staticmethod
    def static_linear(
        x: ivy.Container,
        weight: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.linear. This method simply
        wraps the function, and so the docstring for ivy.linear also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            The input x to compute linear transformation on.
            *[outer_batch_shape,inner_batch_shape,in_features]*
        weight
            The weight matrix. *[outer_batch_shape,out_features,in_features]*
        bias
            The bias vector, default is ``None``. *[outer_batch_shape,out_features]*
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
            optional output array, for writing the result to. It must have a shape 
            that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the linear transformation.
            *[outer_batch_shape,inner_batch_shape,out_features]*

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[1.1, 2.2, 3.3], \
                                           [11., 22., 33.]]), \
                              b=ivy.array([[1.245, 0.278, 4.105], \
                                           [7., 13., 17.]]))
        >>> w = ivy.array([[1., 2., 3.], \
                           [4., 5., 6.], \
                           [7., 8., 9.]])
        >>> b = ivy.array([1., 0., -1.])
        >>> y = ivy.Container.static_linear(x, w, bias=b)
        >>> print(y)
        {
            a: ivy.array([[16.4, 35.2, 54.],
                          [155., 352., 549.]]),
            b: ivy.array([[15.1, 31., 46.9],
                          [85., 195., 305.]])
        }

        >>> x = ivy.Container(a=ivy.array([[1.1, 2.2, 3.3], \
                                           [.0, .1, .2]]), \
                              b=ivy.array([[1.245, 0.278, 4.105], \
                                           [.7, .8, .9]]))
        >>> w = ivy.Container(a=ivy.array([[1., 2., 3.]]), \
                              b=ivy.array([[.1, .2, .3]]))
        >>> b = ivy.Container(a=ivy.array([1.]), b=ivy.array([-1.]))
        >>> y = ivy.Container.static_linear(x, w, bias=b)
        >>> print(y)
        {
            a: ivy.array([[16.4], 
                          [1.8]]),
            b: ivy.array([[0.412], 
                          [-0.5]])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "linear",
            x,
            weight,
            bias=bias,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def linear(
        self: ivy.Container,
        weight: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.linear. This method simply
        wraps the function, and so the docstring for ivy.linear also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input container to compute linear transformation on.
            *[outer_batch_shape,inner_batch_shape,in_features]*
        weight
            The weight matrix. *[outer_batch_shape,out_features,in_features]*
        bias
            The bias vector, default is ``None``. *[outer_batch_shape,out_features]*
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
            optional output array, for writing the result to. It must have a shape 
            that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the linear transformation.
            *[outer_batch_shape,inner_batch_shape,out_features]*

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[1.1, 2.2, 3.3], \
                                           [11., 22., 33.]]), \
                              b=ivy.array([[1.245, 0.278, 4.105], \
                                           [7., 13., 17.]]))
        >>> w = ivy.array([[1., 2., 3.], \
                           [4., 5., 6.], \
                           [7., 8., 9.]])
        >>> b = ivy.array([1, 0, -1])
        >>> y = x.linear(w, bias=b)
        >>> print(y)
        {
            a: ivy.array([[16.4, 35.2, 54.], \
                          [155., 352., 549.]]), \
            b: ivy.array([[15.1, 31., 46.9], \
                          [85., 195., 305.]])
        }
        
        """
        return self.static_linear(
            self,
            weight,
            bias=bias,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_dropout(
        x: ivy.Container,
        prob: float,
        /,
        *,
        scale: bool = True,
        dtype: ivy.Dtype = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "dropout",
            x,
            prob,
            scale=scale,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def dropout(
        self: ivy.Container,
        prob: float,
        /,
        *,
        scale: bool = True,
        dtype: ivy.Dtype = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_dropout(
            self,
            prob,
            scale=scale,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_dropout1d(
        x: ivy.Container,
        prob: float,
        /,
        *,
        training: bool = True,
        data_format: str = "NWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "dropout1d",
            x,
            prob,
            training=training,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def dropout1d(
        self: ivy.Container,
        prob: float,
        /,
        *,
        training: bool = True,
        data_format: str = "NWC",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_dropout1d(
            self,
            prob,
            training=training,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_scaled_dot_product_attention(
        q: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        k: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        v: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        scale: float,
        /,
        *,
        mask: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return ContainerBase.multi_map_in_static_method(
            "scaled_dot_product_attention",
            q,
            k,
            v,
            scale,
            mask=mask,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def scaled_dot_product_attention(
        self: ivy.Container,
        k: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        v: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        scale: float,
        /,
        *,
        mask: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return self.static_scaled_dot_product_attention(
            self,
            k,
            v,
            scale,
            mask=mask,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_multi_head_attention(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        scale,
        num_heads,
        /,
        *,
        context: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        mask: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        to_q_fn: Callable = None,
        to_kv_fn: Callable = None,
        to_out_fn: Callable = None,
        to_q_v=None,
        to_kv_v=None,
        to_out_v=None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return ContainerBase.multi_map_in_static_method(
            "multi_head_attention",
            x,
            scale,
            num_heads,
            context=context,
            mask=mask,
            to_q_fn=to_q_fn,
            to_kv_fn=to_kv_fn,
            to_out_fn=to_out_fn,
            to_q_v=to_q_v,
            to_kv_v=to_kv_v,
            to_out_v=to_out_v,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def multi_head_attention(
        self: ivy.Container,
        scale,
        num_heads,
        /,
        *,
        context: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        mask: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        to_q_fn: Callable = None,
        to_kv_fn: Callable = None,
        to_out_fn: Callable = None,
        to_q_v=None,
        to_kv_v=None,
        to_out_v=None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return self.static_multi_head_attention(
            self,
            scale,
            num_heads,
            context=context,
            mask=mask,
            to_q_fn=to_q_fn,
            to_kv_fn=to_kv_fn,
            to_out_fn=to_out_fn,
            to_q_v=to_q_v,
            to_kv_v=to_kv_v,
            to_out_v=to_out_v,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conv1d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: int,
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        dilations: int = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.conv1d. This method simply
        wraps the function, and so the docstring for ivy.conv1d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,w, d_in]*.
        filters
            Convolution filters *[fw,d_in, d_out]*. (d_in must be the same as d from x)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[2., 3., 4.], [5., 6., 7.]]]),
        ...                   b=ivy.array([[[7., 8., 9.], [10., 11., 12]]]))
        >>> filters = ivy.array([[[0., 0.5, 1.], [0.25, 0.5, 0.75], [-0.5, 0., 0.5 ]]])
        >>> result= ivy.Container.static_conv1d(x,filters,(1,),'VALID')
        >>> print(result)
        {
            ... a: ivy.array([[[-1.25, 2.5, 6.25],
            ...                [-2., 5.5, 13.]]]),
            ... b: ivy.array([[[-2.5, 7.5, 17.5],
            ...                [-3.25, 10.5, 24.2]]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "conv1d",
            x,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conv1d(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: int,
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        dilations: int = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.conv1d. This method simply
        wraps the function, and so the docstring for ivy.conv1d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,w, d_in]*.
        filters
            Convolution filters *[fw,d_in, d_out]*. (d_in must be the same as d from x)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[2., 3., 4.], [5., 6., 7.]]]),
        ...                   b=ivy.array([[[7., 8., 9.], [10., 11., 12]]]))
        >>> filters = ivy.array([[[0., 0.5, 1.], [0.25, 0.5, 0.75], [-0.5, 0., 0.5 ]]])
        >>> result= x.conv1d(filters, (1,), 'VALID')
        >>> print(result)
        {
            ... a: ivy.array([[[-1.25, 2.5, 6.25],
            ...                [-2., 5.5, 13.]]]),
            ... b: ivy.array([[[-2.5, 7.5, 17.5],
            ...                [-3.25, 10.5, 24.2]]])
        }
        """
        return self.static_conv1d(
            self,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conv2d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.conv2d. This method simply
        wraps the function, and so the docstring for ivy.conv2d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d_in]*.
        filters
            Convolution filters *[fh,fw,d_in,d_out]*.
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
            The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
        ...                   b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
        >>> filters = ivy.array([[2., 0., 1.],
        ...                      [1., 3., 1.],
        ...                      [0., 1., 1.]]).reshape((3, 3, 1, 1))
        >>> result = ivy.Container.static_conv2d(x, filters, (2,), 'SAME')
        >>> print(result)
        {
            a:ivy.array([[[[4.],[0.]],[[1.],[5.]]]]),
            b:ivy.array([[[[4.],[0.],[0.]],[[1.],[6.],[0.]],[[0.],[1.],[5.]]]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "conv2d",
            x,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conv2d(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of `ivy.conv2d`. This method simply
        wraps the function, and so the docstring for `ivy.conv2d` also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d_in]*.
        filters
            Convolution filters *[fh,fw,d_in,d_out]*.
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
            The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
        ...                   b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
        >>> filters = ivy.array([[2, 0, 1],
        ...                      [1, 3, 1],
        ...                      [0, 1, 1]], dtype=ivy.float32).reshape((3, 3, 1, 1))
        >>> result = x.conv2d(filters, 2, 'SAME')
        >>> print(result)
        {
            a:ivy.array([[[[4.],[0.]],[[1.],[5.]]]]),
            b:ivy.array([[[[4.],[0.],[0.]],[[1.],[6.],[0.]],[[0.],[1.],[5.]]]])
        }
        """
        return self.static_conv2d(
            self,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conv1d_transpose(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: int,
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        data_format: str = "NWC",
        dilations: int = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return ContainerBase.multi_map_in_static_method(
            "conv1d_transpose",
            x,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conv1d_transpose(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: int,
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        data_format: str = "NWC",
        dilations: int = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return self.static_conv1d_transpose(
            self,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conv2d_transpose(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        data_format: str = "NHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return ContainerBase.multi_map_in_static_method(
            "conv2d_transpose",
            x,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conv2d_transpose(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        data_format: str = "NHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray, ivy.Container]:
        return self.static_conv2d_transpose(
            self,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_depthwise_conv2d(
        x: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.depthwise_conv2d. This method simply
        wraps the function, and so the docstring for ivy.depthwise_conv2d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d]*.
        filters
            Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from x)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> a = ivy.randint(0, 255, shape=(1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> b = ivy.randint(0, 255, shape=(1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> inp = ivy.Container(a=a, b=b)
        >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3])
        >>> y = ivy.Container.static_depthwise_conv2d(
        ...                                            inp,
        ...                                            filters,
        ...                                            strides=2,
        ...                                            padding='SAME')
        >>> print(y.shape)
        [1, 64, 64, 3]
        """
        return ContainerBase.multi_map_in_static_method(
            "depthwise_conv2d",
            x,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def depthwise_conv2d(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.depthwise_conv2d. This method
        simply wraps the function, and so the docstring for ivy.depthwise_conv2d
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d]*.
        filters
            Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from self)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> a = ivy.randint(0, 255, shape=(1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> b = ivy.randint(0, 255, shape=(1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> inp = ivy.Container(a=a, b=b)
        >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3])
        >>> y = inp.depthwise_conv2d(filters, 2, 'SAME')
        >>> print(y.shape)
        [1, 64, 64, 3]
        """
        return self.static_depthwise_conv2d(
            self,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conv3d(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.conv3d. This method simply
        wraps the function, and so the docstring for ivy.conv3d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        filters
            Convolution filters *[fdfh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.full((1, 2, 3, 3, 1),0.5),\
                              b = ivy.full((1, 2, 5, 5, 1),1.))

        >>> filters = ivy.ones((3, 3, 3, 1, 1))

        >>> result = ivy.Container.static_conv3d(x, filters, 2, 'SAME')
        >>> print(result)
        {
            a: ivy.array([[[[[4.],[4.]],[[4.],[4.]]]]]),
            b: ivy.array([[[[[8.],[12.],[8.]],[[12.],[18.],[12.]],[[8.],[12.],[8.]]]]])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "conv3d",
            x,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conv3d(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.conv3d. This method simply
        wraps the function, and so the docstring for ivy.conv3d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        filters
            Convolution filters *[fdfh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NDHWC" or "NCDHW". Defaults to "NDHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.full((1, 2, 3, 3, 1),0.5),\
                              b = ivy.full((1, 2, 5, 5, 1),1.))

        >>> filters = ivy.ones((3, 3, 3, 1, 1))

        >>> result = x.conv3d(filters, 2, 'SAME')
        >>> print(result)
        {
            a: ivy.array([[[[[4.],[4.]],[[4.],[4.]]]]]),
            b: ivy.array([[[[[8.],[12.],[8.]],[[12.],[18.],[12.]],[[8.],[12.],[8.]]]]])
        }

        """
        return self.static_conv3d(
            self,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conv3d_transpose(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        output_shape: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        data_format: str = "NDHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "conv3d_transpose",
            x,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conv3d_transpose(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        output_shape: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        data_format: str = "NDHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_conv3d_transpose(
            self,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_lstm_update(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        init_h: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        init_c: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        recurrent_kernel: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        bias: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        recurrent_bias: Optional[
            Union[ivy.Array, ivy.NativeArray, ivy.Container]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Tuple[ivy.Container, ivy.Container]:
        return ContainerBase.multi_map_in_static_method(
            "lstm_update",
            x,
            init_h,
            init_c,
            kernel,
            recurrent_kernel,
            bias=bias,
            recurrent_bias=recurrent_bias,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lstm_update(
        self: ivy.Container,
        init_h: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        init_c: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        kernel: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        recurrent_kernel: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        bias: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
        recurrent_bias: Optional[
            Union[ivy.Array, ivy.NativeArray, ivy.Container]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Tuple[ivy.Container, ivy.Container]:
        return self.static_lstm_update(
            self,
            init_h,
            init_c,
            kernel,
            recurrent_kernel,
            bias=bias,
            recurrent_bias=recurrent_bias,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
