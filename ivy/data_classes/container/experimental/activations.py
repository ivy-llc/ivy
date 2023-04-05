# global
from typing import Union, Optional, List, Dict

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithActivationExperimental(ContainerBase):
    @staticmethod
    def static_logit(
        x: Union[float, int, ivy.Container],
        /,
        *,
        eps: Optional[float] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logit.
        This method simply wraps the function, and so the
        docstring for ivy.logit  also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input container.
        eps
            When eps is None the function outpus NaN where x < 0 or x > 1.
            and inf or -inf where x = 1 or x = 0, respectively.
            Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
        out
            Optional output Contaner.

        Returns
        -------
        ret
            Container with logits of the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, 0.9])
        >>> b = ivy.array([0.1, 2, -0.9])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_logit(x)
        >>> print(z)
        {
            a: ivy.array([inf, -inf, 2.19722438]),
            b: ivy.array([-2.19722462, nan, nan])
        }

        >>> a = ivy.array([0.3, 2, 0.9])
        >>> b = ivy.array([0.1, 1.2, -0.9])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_logit(x, eps=0.2)
        >>> print(z)
        {
            a: ivy.array([-0.84729779, 1.38629448, 1.38629448]),
            b: ivy.array([-1.38629436, 1.38629448, -1.38629436])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logit",
            x,
            eps=eps,
            out=out,
        )

    def logit(
        self: Union[float, int, ivy.Container],
        /,
        *,
        eps: Optional[float] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logit.
        This method simply wraps the function, and so the
        docstring for ivy.logit  also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container.
        eps
            When eps is None the function outpus NaN where x < 0 or x > 1.
            and inf or -inf where x = 1 or x = 0, respectively.
            Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
        out
            Optional output Contaner.

        Returns
        -------
        ret
            Container with logits of the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, 0.9])
        >>> b = ivy.array([0.1, 2, -0.9])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = x.logit()
        >>> print(z)
        {
            a: ivy.array([inf, -inf, 2.19722438]),
            b: ivy.array([-2.19722462, nan, nan])
        }

        >>> a = ivy.array([0.3, 2, 0.9])
        >>> b = ivy.array([0.1, 1.2, -0.9])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = x.logit(eps=0.2)
        >>> print(z)
        {
            a: ivy.array([-0.84729779, 1.38629448, 1.38629448]),
            b: ivy.array([-1.38629436, 1.38629448, -1.38629436])
        }

        """
        return self.static_logit(self, eps=eps, out=out)

    @staticmethod
    def static_hardshrink(
        x: Union[float, int, ivy.Container],
        /,
        *,
        lambd: Optional[float] = 0.5,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hardshrink.
        This method simply wraps the function, and so the
        docstring for ivy.hardshrink also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input container.
        lambd
            The value where the function is zero for inputs that are absolute value
            less than it.
        out
            Optional output container.

        Returns
        -------
        ret
            Container with hardshrink of the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, -0.9])
        >>> b = ivy.array([0.1, 2, -0.4])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_hardshrink(x)
        >>> print(z)
        {
            a: ivy.array([1., 0., -0.9]),
            b: ivy.array([0., 2., 0.])
        }

        >>> a = ivy.array([0.3, -0.3, 0.1])
        >>> b = ivy.array([-0.1, -1.2, -0.1])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_hardshrink(x, lambd=0.2)
        >>> print(z)
        {
            a: ivy.array([0.3, -0.3, 0.]),
            b: ivy.array([0., -1.2, 0.])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "hardshrink",
            x,
            lambd=lambd,
            out=out,
        )

    def hardshrink(
        self: Union[float, int, ivy.Container],
        /,
        *,
        lambd: Optional[float] = 0.5,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.hardshrink.
        This method simply wraps the function, and so the
        docstring for ivy.hardshrink  also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container.
        lambd
            The value where the function is zero for inputs that are absolute value
            less than it.
        out
            Optional output container.

        Returns
        -------
        ret
            Container with hardshrink of the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, -0.9])
        >>> b = ivy.array([0.1, 2, -0.4])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = x.hardshrink()
        >>> print(z)
        {
            a: ivy.array([1., 0., -0.9]),
            b: ivy.array([0., 2., 0.])
        }

        >>> a = ivy.array([0.3, -0.3, 0.1])
        >>> b = ivy.array([-0.1, -1.2, -0.1])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = x.hardshrink(lambd=0.2)
        >>> print(z)
        {
            a: ivy.array([0.3, -0.3, 0.]),
            b: ivy.array([0., -1.2, 0.])
        }

        """
        return self.static_hardshrink(self, lambd=lambd, out=out)

    @staticmethod
    def static_softshrink(
        x: Union[float, int, ivy.Container],
        /,
        *,
        lambd: Optional[float] = 0.5,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.softshrink.
        This method simply wraps the function, and so the
        docstring for ivy.softshrink also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input container.
        lambd
            The value where the function is zero for inputs that are absolute value
            less than it. It must be no less than zero.
        out
            Optional output container.

        Returns
        -------
        ret
            Container with softshrink of the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, -0.9])
        >>> b = ivy.array([0.1, 2, -0.4])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_softshrink(x)
        >>> print(z)
        {
            a: ivy.array([0.5, 0., -0.4]),
            b: ivy.array([0., 1.5, 0.])
        }

        >>> a = ivy.array([0.3, -0.3, 0.1])
        >>> b = ivy.array([-0.1, -1.2, -0.1])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_softshrink(x, lambd=0.2)
        >>> print(z)
        {
            a: ivy.array([0.1, -0.1, 0.]),
            b: ivy.array([0., -1., 0.])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "softshrink",
            x,
            lambd=lambd,
            out=out,
        )

    def softshrink(
        self: Union[float, int, ivy.Container],
        /,
        *,
        lambd: Optional[float] = 0.5,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softshrink.
        This method simply wraps the function, and so the
        docstring for ivy.softshrink  also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container.
        lambd
            The value where the function is zero for inputs that are absolute value
            less than it. It must be no less than zero.
        out
            Optional output container.

        Returns
        -------
        ret
            Container with softshrink of the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, -0.9])
        >>> b = ivy.array([0.1, 2, -0.4])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = x.softshrink()
        >>> print(z)
        {
            a: ivy.array([0.5, 0., -0.4]),
            b: ivy.array([0., 1.5, 0.])
        }

        >>> a = ivy.array([0.3, -0.3, 0.1])
        >>> b = ivy.array([-0.1, -1.2, -0.1])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = x.softshrink(lambd=0.2)
        >>> print(z)
        {
            a: ivy.array([0.1, -0.1, 0.]),
            b: ivy.array([0., -1., 0.])
        }

        """
        return self.static_softshrink(self, lambd=lambd, out=out)

    @staticmethod
    def static_thresholded_relu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        threshold: Union[int, float] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.thresholded_relu.
        This method simply wraps the function, and so the docstring
        for ivy.thresholded_relu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        threshold
            threshold value above which the activation is linear. Default: ``0``.
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
            a container with the rectified linear activation unit function
            applied element-wise with custom threshold.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container.static_thresholded_relu(x, threshold=0.5)
        >>> print(y)
        {
            a: ivy.array([1., 0.]),
            b: ivy.array([0., 0.])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "thresholded_relu",
            x,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def thresholded_relu(
        self: ivy.Container,
        /,
        *,
        threshold: Union[int, float] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.thresholded_relu.
        This method simply wraps the function, and so the docstring
        for ivy.thresholded_relu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        threshold
            threshold value above which the activation is linear. Default: ``0``.
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
            a container with the rectified linear activation unit function
            applied element-wise with custom threshold.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.thresholded_relu(threshold=0.5)
        >>> print(y)
        {
            a: ivy.array([1., 0.]),
            b: ivy.array([0., 0.])
        }

        """
        return self.static_thresholded_relu(
            self,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_threshold(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        threshold: Union[int, float],
        value: Union[int, float],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.threshold.
        This method simply wraps the function, and so the docstring
        for ivy.threshold also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        threshold
            threshold value above which the activation is linear.
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
            a container with threshold applied element-wise with custom threshold.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container.static_threshold(x, 0.5, 2.)
        >>> print(y)
        {
            a: ivy.array([1., 2.]),
            b: ivy.array([2., 2.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "threshold",
            x,
            threshold,
            value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def threshold(
        self: ivy.Container,
        threshold: Union[int, float],
        value: Union[int, float],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.threshold.
        This method simply wraps the function, and so the docstring
        for ivy.threshold also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        threshold
            threshold value above which the activation is linear.
        value
            the value to replace with
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
            a container with threshold applied element-wise with custom threshold.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.threshold(0.5, 2.)
        >>> print(y)
        {
            a: ivy.array([1., 2.]),
            b: ivy.array([2., 2.])
        }
        """
        return self.static_threshold(
            self,
            threshold,
            value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_relu6(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.relu6.
        This method simply wraps the function, and so the docstring
        for ivy.relu6 also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
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
            a container with the rectified linear 6 activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = {
                    a: ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
                    b: ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
                }
        >>> y = ivy.Container.static_relu6(x)
        >>> print(y)
        {
            a: ivy.array([0., 0., 0., 0., 1., 2., 3., 4., 5.]),
            b: ivy.array([1., 2., 3., 4., 5., 6., 6., 6., 6.])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "relu6",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def relu6(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.relu6.
        This method simply wraps the function, and so the docstring
        for ivy.relu6 also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
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
            a container with the rectified linear 6 activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = {
                    a: ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
                    b: ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
                }
        >>> y = x.relu()
        >>> print(y)
        {
            a: ivy.array([0., 0., 0., 0., 1., 2., 3., 4., 5.]),
            b: ivy.array([1., 2., 3., 4., 5., 6., 6., 6., 6.])
        }

        """
        return self.static_relu6(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_batch_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        variance: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: bool = False,
        eps: float = 1e-5,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.batch_norm.
        This method simply wraps the function, and so the docstring
        for ivy.batch_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        mean
            A mean array for the input's normalization.
        variance
            A variance array for the input's normalization.
        offset
            An offset array. If present, will be added to the normalized input.
        scale
            A scale array. If present, the scale is applied to the normalized input.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
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
             Container containing the normalized, scaled, offset values.
        """
        return ContainerBase.cont_multi_map_in_function(
            "batch_norm",
            x,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def batch_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        variance: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: bool = False,
        eps: float = 1e-5,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.batch_norm.
        This method simply wraps the function, and so the docstring
        for ivy.batch_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        mean
            A mean array for the input's normalization.
        variance
            A variance array for the input's normalization.
        offset
            An offset array. If present, will be added to the normalized input.
        scale
            A scale array. If present, the scale is applied to the normalized input.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
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
             Container containing the normalized, scaled, offset values.
        """
        return self.static_batch_norm(
            self,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_group_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        num_groups: int,
        /,
        *,
        weight: Union[ivy.NativeArray, ivy.Array, ivy.Container] = None,
        bias: Union[ivy.NativeArray, ivy.Array, ivy.Container] = None,
        eps: float = 1e-5,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.group_norm.
        This method simply wraps the function, and so the docstring
        for ivy.group_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        num_groups
            Number of groups to separate the channels into
        weight
            A scale array. If present, the scale is applied to the normalized input.
        bias
            An offset array. If present, will be added to the normalized input.
        eps
            A small float number to avoid dividing by 0.
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
             Container containing the normalized, scaled, offset values.
        """
        return ContainerBase.cont_multi_map_in_function(
            "group_norm",
            x,
            num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def group_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        num_groups: int,
        /,
        *,
        weight: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        bias: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        eps: float = 1e-5,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.group_norm.
        This method simply wraps the function, and so the docstring
        for ivy.group_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        num_groups
            Number of groups to separate the channels into
        weight
            A scale array. If present, the scale is applied to the normalized input.
        bias
            An offset array. If present, will be added to the normalized input.
        eps
            A small float number to avoid dividing by 0.
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
             Container containing the normalized, scaled, offset values.
        """
        return self.static_group_norm(
            self,
            num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_logsigmoid(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logsigmoid.
        This method simply wraps the function, and so the
        docstring for ivy.logsigmoid also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            Input container.
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
            Container with Log-sigmoid applied to the leaves.

        Examples
        --------
        >>> a = ivy.array([1, 0, 0.9])
        >>> b = ivy.array([0.1, 2, -0.9])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_logsigmoid(x)
        >>> print(z)
        {
            a: ivy.array([-0.31326169, -0.69314718, -0.34115386]),
            b: ivy.array([-0.64439666, -0.126928, -1.24115384])
        }

        >>> a = ivy.array([0.3, 2.5, 4.9])
        >>> b = ivy.array([0.1, 1.2, -9.])
        >>> x = ivy.Container(a=a, b=b)
        >>> z = ivy.Container.static_logsigmoid(x)
        >>> print(z)
        {
            a: ivy.array([-0.55435526, -0.07888974, -0.00741899]),
            b: ivy.array([-0.64439666, -0.26328245, -9.00012302])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logsigmoid",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logsigmoid(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Applies element-wise Log-sigmoid of x i.e. log(1 / (1 + exp(-x)).

        Parameters
        ----------
        self
            Input container.

        Returns
        -------
        ret
            Container with Log-sigmoid applied to the leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.logsigmoid()
        >>> print(y)
        {
            a: ivy.array([-0.31326163, -1.46328258]),
            b: ivy.array([-0.51301527, -0.79813886])
        }

        """
        return self.static_logsigmoid(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_selu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.selu.
        This method simply wraps the function, and so the docstring
        for ivy.selu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the selu unit function applied element-wise.

        Examples
        --------
        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = ivy.Container.static_selu(x)
        >>> print(y)

        {
            a: ivy.array([-1.11,  1.05 ,  2.1]),
            b: ivy.array([ 0.52,  0., -0.17])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "selu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def selu(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.selu.
        This method simply wraps the function, and so the docstring
        for ivy.selu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the selu function applied element-wise.

        Examples
        --------

        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = x.selu()
        >>> print(y)
        {
            a: ivy.array([-1.11,  1.05 ,  2.1]),
            b: ivy.array([ 0.52,  0., -0.17])
        }
        """
        return self.static_selu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_hard_tanh(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        min_value: float = -1.0,
        max_value: float = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hard_tanh.
        This method simply wraps the function, and so the docstring
        for ivy.hard_tanh also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        min_val
            Minimum value of the linear region range.
        max_val
            Maximum value of the linear region range
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the hard tanh function applied element-wise.

        Examples
        --------
        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = ivy.Container.static_hard_tanh(x)
        >>> print(y)
        {
            a: ivy.array([-1.,  1.,  1.]),
            b: ivy.array([ 0.5,  0. , -0.1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hard_tanh",
            x,
            min_value=min_value,
            max_value=max_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hard_tanh(
        self: ivy.Container,
        /,
        *,
        min_value: float = -1.0,
        max_value: float = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.hard_tanh.
        This method simply wraps the function, and so the docstring
        for ivy.hard_tanh also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        min_val
            Minimum value of the linear region range.
        max_val
            Maximum value of the linear region range
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the hard tanh function applied element-wise.

        Examples
        --------
        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = x.hard_tanh()
        >>> print(y)
        {
            a: ivy.array([-1.,  1.,  1.]),
            b: ivy.array([ 0.5,  0. , -0.1])
        }
        """
        return self.static_hard_tanh(
            self,
            min_value=min_value,
            max_value=max_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_hard_sigmoid(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hard_sigmoid.
        This method simply wraps the function, and so the docstring
        for ivy.hard_sigmoid also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the hard sigmoid unit function applied element-wise.

        Examples
        --------
        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = ivy.Container.static_hard_sigmoid(x)
        >>> print(y)
        {
            a: ivy.array([0.33, 0.67, 0.83]),
            b: ivy.array([0.58, 0.5, 0.48])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hard_sigmoid",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hard_sigmoid(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.hard_sigmoid.
        This method simply wraps the function, and so the docstring
        for ivy.hard_sigmoid also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
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
            a container with the sigmoid unit function applied element-wise.

        Examples
        --------
        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = x.hard_sigmoid()
        >>> print(y)
        {
            a: ivy.array([0.33, 0.67, 0.83]),
            b: ivy.array([0.58, 0.5, 0.48])
        }
        """
        return self.static_hard_sigmoid(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_softsign(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.softsign.
        This method simply wraps the function, and so the docstring
        for ivy.softsign also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the softsign function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = ivy.Container.static_softsign(x)
        >>> print(y)
        {
            a: ivy.array([-0.26, -0.39]),
            b: ivy.array([0.5, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "softsign",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def softsign(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softsign.
        This method simply wraps the function, and so the docstring
        for ivy.softsign also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        threshold
            values above this revert to a linear function. Default: ``None``.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the softplus unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = ivy.x.static_softsign()
        >>> print(y)
        {
            a: ivy.array([-0.26, -0.39]),
            b: ivy.array([0.5, 0.])
        }
        """
        return self.static_softsign(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_silu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.silu.
        This method simply wraps the function, and so the docstring
        for ivy.silu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the silu function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = ivy.Container.static_silu(x)
        >>> print(y)
        {
            a: ivy.array([-0.14, -0.22]),
            b: ivy.array([0.7310586, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "silu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def silu(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.silu.
        This method simply wraps the function, and so the docstring
        for ivy.silu also applies to this method with minimal changes.
        Parameters
        ----------
        self
            Input container.
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
            a container with the silu function applied element-wise.
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = x.silu()
        >>> print(y)
        {
            a: ivy.array([-0.14, -0.22]),
            b: ivy.array([0.7310586, 0.])
        }
        """
        return self.static_silu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_hard_silu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hard_silu.
        This method simply wraps the function, and so the docstring
        for ivy.hard_silu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the hard silu function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = ivy.Container.static_hard_silu(x)
        >>> print(y)
        {
            a: ivy.array([-0.15, -0.25])
            b: ivy.array([0.67, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hard_silu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hard_silu(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softplus.
        This method simply wraps the function, and so the docstring
        for ivy.softplus also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        beta
            The beta value for the softplus formation. Default: ``None``.
        threshold
            Values above this revert to a linear function. Default: ``None``.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the hard silu function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = x.hard_silu()
        >>> print(y)
        {
            a: ivy.array([-0.15, -0.25])
            b: ivy.array([0.67, 0.])
        }
        """
        return self.static_hard_silu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_elu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.elu.
        This method simply wraps the function, and so the docstring
        for ivy.elu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        alpha
            Array or scalar specifying the negative slope.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elu function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_elu(x)
        >>> print(y)
        {
            a: ivy.array([ 0.39, -0.57]),
            b: ivy.array([ 1., -0.18])
        }

        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_elu(x, alpha=0.1)
        >>> print(y)
        {
            a: ivy.array([ 0.39, -0.057]),
            b: ivy.array([ 1., -0.02])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "elu",
            x,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def elu(
        self: ivy.Container,
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.elu.
        This method simply wraps the function, and so the docstring
        for ivy.elu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        alpha
            Array or scalar specifying the negative slope.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
           A container with the elu function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.elu()
        >>> print(y)
        {
            a: ivy.array([ 0.39, -0.57]),
            b: ivy.array([ 1., -0.18])
        }

        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.elu(alpha=0.1)
        >>> print(y)
        {
            a: ivy.array([ 0.39, -0.057]),
            b: ivy.array([ 1., -0.02])
        }

        """
        return self.static_elu(
            self,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_parametric_relu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        weight: Union[float, ivy.Array],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.parametric_relu.
        This method simply wraps the function, and so the docstring
        for ivy.parametric_relu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        weight
            Array or scalar specifying the negative slope.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the parametric relu unit function applied element-wise.
        """
        return ContainerBase.cont_multi_map_in_function(
            "parametric_relu",
            x,
            weight,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def parametric_relu(
        self: ivy.Container,
        weight: Union[float, ivy.Array],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.parametric_relu.
        This method simply wraps the function, and so the docstring
        for ivy.parametric_relu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        weight
            Array or scalar specifying the negative slope.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
           A container with the parmaetric relu function applied element-wise.

        """
        return self.static_parametric_relu(
            self,
            weight,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_celu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.celu.
        This method simply wraps the function, and so the docstring
        for ivy.celu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        alpha
            Array or scalar specifying the negative slope.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the celu function applied element-wise.

        Examples
        --------
        >>> x = x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_celu(x)
        >>> print(y)
        {
            a: ivy.array([0.39, -0.57]),
            b: ivy.array([1., -0.18])
        }


        >>> x = x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_celu(x, alpha=2.0)
        >>> print(y)
        {
            a: ivy.array([0.39, -0.69]),
            b: ivy.array([1., -0.19])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "celu",
            x,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def celu(
        self: ivy.Container,
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.celu.
        This method simply wraps the function, and so the docstring
        for ivy.celu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        alpha
            Array or scalar specifying the negative slope.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
           A container with the celu function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.celu()
        >>> print(y)
        {
            a: ivy.array([0.39, -0.57]),
            b: ivy.array([1., -0.18])
        }

        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.celu(alpha=2.0)
        >>> print(y)
        {
            a: ivy.array([0.39, -0.69]),
            b: ivy.array([1., -0.19])
        }

        """
        return self.static_celu(
            self,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_glu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.glu.
        This method simply wraps the function, and so the docstring
        for ivy.glu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        axis
            The axis along which the split should be computed
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the glu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_glu(x)
        >>> print(y)
        {
            a: ivy.array([0.12]),
            b: ivy.array([0.45])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "glu",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def glu(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.glu. This method simply wraps the
        function, and so the docstring for ivy.glu also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container.
        axis
            The axis along which the split should be computed
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the glu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.glu(x)
        >>> print(y)
        {
            a: ivy.array([0.12]),
            b: ivy.array([0.45])
        }
        """

        return self.static_glu(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_selu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.selu.
        This method simply wraps the function, and so the docstring
        for ivy.selu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
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
            a container with the scaled exponential linear unit activation function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container.static_selu(x)
        >>> print(y)
        {
            a: ivy.array([1.05070102, -1.22856998]),
            b: ivy.array([0.42028043, -0.31868932])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "selu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def selu(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.selu.
        This method simply wraps the function, and so the docstring
        for ivy.selu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
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
            a container with the scaled exponential linear unit activation function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.selu()
        >>> print(y)
        {
            a: ivy.array([1.05070102, -1.22856998]),
            b: ivy.array([0.42028043, -0.31868932])
        }

        """
        return self.static_selu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
