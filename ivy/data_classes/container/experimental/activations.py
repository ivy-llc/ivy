# global
from typing import Union, Optional, List, Dict, Literal

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithActivationExperimental(ContainerBase):
    @staticmethod
    def static_logit(
        x: Union[float, int, ivy.Container],
        /,
        *,
        eps: Optional[Union[float, ivy.Container]] = None,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.logit. This method simply
        wraps the function, and so the docstring for ivy.logit  also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        eps
            When eps is None the function outputs NaN where x < 0 or x > 1.
            and inf or -inf where x = 1 or x = 0, respectively.
            Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            Optional output Container.

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
            complex_mode=complex_mode,
            out=out,
        )

    def logit(
        self: Union[float, int, ivy.Container],
        /,
        *,
        eps: Optional[Union[float, ivy.Container]] = None,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.logit. This method
        simply wraps the function, and so the docstring for ivy.logit  also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        eps
            When eps is None the function outputs NaN where x < 0 or x > 1.
            and inf or -inf where x = 1 or x = 0, respectively.
            Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            Optional output Container.

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
        return self.static_logit(self, eps=eps, complex_mode=complex_mode, out=out)

    @staticmethod
    def static_thresholded_relu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        threshold: Union[int, float, ivy.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.thresholded_relu. This
        method simply wraps the function, and so the docstring for
        ivy.thresholded_relu also applies to this method with minimal changes.

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
        threshold: Union[int, float, ivy.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.thresholded_relu. This
        method simply wraps the function, and so the docstring for
        ivy.thresholded_relu also applies to this method with minimal changes.

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
    def static_prelu(
        x: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        slope: Union[float, ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """

        Parameters
        ----------
        x
        slope
        key_chains
        to_apply
        prune_unapplied
        map_sequences
        out
        """
        return ContainerBase.cont_multi_map_in_function(
            "prelu",
            x,
            slope,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def prelu(
        self: ivy.Container,
        slope: Union[float, ivy.NativeArray, ivy.Array, ivy.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """

        Parameters
        ----------
        slope
        key_chains
        to_apply
        prune_unapplied
        map_sequences
        out
        """
        return self.static_prelu(
            self,
            slope,
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
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.relu6. This method simply
        wraps the function, and so the docstring for ivy.relu6 also applies to
        this method with minimal changes.

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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
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
        >>> x = ivy.Container(a = ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        ...                   b = ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]))
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
            complex_mode=complex_mode,
            out=out,
        )

    def relu6(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.relu6. This method
        simply wraps the function, and so the docstring for ivy.relu6 also
        applies to this method with minimal changes.

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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
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
        >>> x = ivy.Container(a = ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        ...                   b= ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]))
        >>> y = x.relu()
        >>> print(y)
        {
            a: ivy.array([0., 0., 0., 0., 1., 2., 3., 4., 5.]),
            b: ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        }
        """
        return self.static_relu6(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            complex_mode=complex_mode,
            out=out,
        )

    @staticmethod
    def static_logsigmoid(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.logsigmoid. This method
        simply wraps the function, and so the docstring for ivy.logsigmoid also
        applies to this method with minimal changes.

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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.

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
            input,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            complex_mode=complex_mode,
        )

    def logsigmoid(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ) -> ivy.Container:
        """Apply element-wise Log-sigmoid of x i.e. log(1 / (1 + exp(-x)).

        Parameters
        ----------
        self
            Input container.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.

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
            complex_mode=complex_mode,
        )

    @staticmethod
    def static_selu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.selu. This method simply
        wraps the function, and so the docstring for ivy.selu also applies to
        this method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.selu. This method
        simply wraps the function, and so the docstring for ivy.selu also
        applies to this method with minimal changes.

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

    @staticmethod
    def _static_silu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.silu. This method simply
        wraps the function, and so the docstring for ivy.silu also applies to
        this method with minimal changes.

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
            a container with the rectified linear activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container.static_silu(x)
        >>> print(y)
        {
            a: ivy.array([0.73105854, -0.27777028]),
            b: ivy.array([0.23947507, -0.0900332])
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
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.silu. This method
        simply wraps the function, and so the docstring for ivy.silu also
        applies to this method with minimal changes.

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
            a container with the rectified linear activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.silu()
        >>> print(y)
        {
            a: ivy.array([0.73105854, -0.27777028]),
            b: ivy.array([0.23947507, -0.0900332])
        }
        """
        return self._static_silu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_elu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.elu. This method simply
        wraps the function, and so the docstring for ivy.elu also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        alpha
            scaler for controlling the slope of the function for x <= 0 Default: 1.0
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
             a container with the elu unit function applied element-wise.

        Examples
        --------
        >>> x = x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_elu(x)
        >>> print(y)
        {
            a: ivy.array([0.38999999, -0.57]),
            b: ivy.array([1., -0.18])
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
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.elu. This method simply
        wraps the function, and so the docstring for ivy.elu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        alpha
            scaler for controlling the slope of the function for x <= 0 Default: 1.0
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
           a container with the elu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.elu()
        >>> print(y)
        {
            a: ivy.array([0.38999999, -0.57]),
            b: ivy.array([1., -0.18])
        }
        """
        return self._static_elu(
            self,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_hardtanh(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        min_val: ivy.Container = -1.0,
        max_val: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.hardtanh.This method
        simply wrap the function,the docstring for ivy.hardtanh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        min_val
             minimum value of the linear region range. Default: -1.
        max_val
            maximum value of the linear region range. Default: 1.
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
             a container with the hardtanh unit function applied element-wise.

        Examples
        --------
        >>> x = x = ivy.Container(a=ivy.array([0.39, -2.0]), b=ivy.array([2., -0.2]))
        >>> y = ivy.Container._static_hardtanh(x)
        >>> print(y)
        {
            a: ivy.array([0.3899, -1.]),
            b: ivy.array([1., -0.2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hardtanh",
            x,
            min_val=min_val,
            max_val=max_val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hardtanh(
        self: ivy.Container,
        /,
        *,
        min_val: ivy.Container = -1.0,
        max_val: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.hardtanh.This method
        simply wraps the function, so the docstring for ivy.elu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        min_val
             minimum value of the linear region range. Default: -1.
        max_val
            maximum value of the linear region range. Default: 1.
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
           a container with the hardtanh unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -2.0]), b=ivy.array([2., -0.2]))
        >>> y = ivy.Container.hardtanh(x)
        >>> print(y)
        {
            a: ivy.array([0.389999, -1.]),
            b: ivy.array([1., -0.2])
        }
        """
        return self._static_hardtanh(
            self,
            max_val=max_val,
            min_val=min_val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tanhshrink(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.tanhshrink. This method
        simply wraps the function, and so the docstring for ivy.tanhshrink also
        applies to this method with minimal changes.

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
            a container with the tanhshrink activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container._static_tanhshrink(x)
        >>> print(y)
        {
            a: ivy.array([0.23840582, -0.36634541]),
            b: ivy.array([0.02005103, -0.00262468])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "tanhshrink",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tanhshrink(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.tanhshrink. This method
        simply wraps the function, and so the docstring for ivy.tanhshrink also
        applies to this method with minimal changes.

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
            a container with the tanhshrink activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.tanhshrink()
        >>> print(y)
        {
            a: ivy.array([0.23840582, -0.36634541]),
            b: ivy.array([0.02005103, -0.00262468])
        }
        """
        return self._static_tanhshrink(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_threshold(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        threshold: ivy.Container,
        value: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.threshold. This method
        simply wraps the function, and so the docstring for ivy.threshold also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        threshold
            threshold value for thresholding operation.
        value
            value to replace with if thresholding condition is not met.
        key_chains
            The key-chains to apply or not apply the method to.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the threshold activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x._static_threshold(threshold=0.5, value=0.0)
        >>> print(y)
        {
            a: ivy.array([1., 0.]),
            b: ivy.array([0., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "threshold",
            x,
            threshold=threshold,
            value=value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def threshold(
        self: ivy.Container,
        /,
        *,
        threshold: ivy.Container,
        value: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.threshold. This method
        simply wraps the function, and so the docstring for ivy.threshold also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        threshold
            threshold value for thresholding operation.
        value
            value to replace with if thresholding condition is not met.
        key_chains
            The key-chains to apply or not apply the method to.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the threshold activation unit function
            applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
        >>> y = x.threshold(threshold=0.5, value=0.0)
        >>> print(y)
        {
            a: ivy.array([1., 0.]),
            b: ivy.array([0., 0.])
        }
        """
        return self._static_threshold(
            self,
            threshold=threshold,
            value=value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_softshrink(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        lambd: ivy.Container = 0.5,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = False,
        prune_unapplied: Union[bool, ivy.Container] = True,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.softshrink. This method
        simply wraps the function, and so the docstring for ivy.softshrink also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        lambd
            Lambda value for soft shrinkage calculation.
        key_chains
            The key-chains to apply or not apply the method to.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container with soft shrinkage applied to the leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., -2.]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container._static_softshrink(x)
        >>> print(y)
        {
            a: ivy.array([0.5, -1.5]),
            b: ivy.array([0., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "softshrink",
            x,
            lambd=lambd,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def softshrink(
        self: ivy.Container,
        /,
        *,
        lambd: ivy.Container = 0.5,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = False,
        prune_unapplied: Union[bool, ivy.Container] = True,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Apply the soft shrinkage function element-wise.

        Parameters
        ----------
        self
            Input container.
        lambd
            Lambda value for soft shrinkage calculation.
        key_chains
            The key-chains to apply or not apply the method to.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container with soft shrinkage applied to the leaves.

        Examples
        --------
        >>> import ivy.numpy as np
        >>> x = ivy.Container(a=np.array([1., -2.]), b=np.array([0.4, -0.2]))
        >>> y = ivy.Container.softshrink(x)
        >>> print(y)
        {
            a: ivy.array([0.5, -1.5]),
            b: ivy.array([0., 0.])
        }
        """
        return self._static_softshrink(
            self,
            lambd=lambd,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_celu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.celu. This method simply
        wraps the function, and so the docstring for ivy.celu also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        alpha
            array or scalar specifying the alpha value for CELU formlation.
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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
             a container with the celu unit function applied element-wise.

        Examples
        --------
        >>> x = x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_celu(x)
        >>> print(y)
        {
            a: ivy.array([0.38999999, -0.17]),
            b: ivy.array([1., -0.04])
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
            complex_mode=complex_mode,
            out=out,
        )

    def celu(
        self: ivy.Container,
        /,
        *,
        alpha: ivy.Container = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.leaky_relu. This method
        simply wraps the function, and so the docstring for ivy.leaky_relu also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        alpha
            array or scalar specifying alpha (negative slope) value for CELU
            formulation.
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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
           a container with the celu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.celu()
        >>> print(y)
        {
            a: ivy.array([0.38999999, -0.57]),
            b: ivy.array([1., -0.18])
        }
        """
        return self._static_celu(
            self,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            complex_mode=complex_mode,
            out=out,
        )

    @staticmethod
    def _static_scaled_tanh(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: Union[float, ivy.Container] = 1.7159,
        beta: Union[float, ivy.Container] = 0.67,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.scaled_tanh. This method
        simply wraps the function, and so the docstring for ivy.scaled_tanh
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        alpha
            The scaling parameter for the output.
            Determines the amplitude of the tanh function.
            Default: 1.7159
        beta
            The scaling parameter for the input.
            Determines the slope of the tanh function.
            Default: 0.67
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
             a container with the scaled_tanh function applied.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([8.931, -0.85]), b=ivy.array([1., -0.2])))
        >>> y = ivy.Container._static_scaled_tanh(x)
        >>> y
        {
            a: ivy.array([1.71587813, -0.88367474]),
            b: ivy.array([1.00376701, -0.2285642])
        }

        >>> x = ivy.Container(a=ivy.array([8.9, -8.9]), b=ivy.array([3., 33.2]))
        >>> y = ivy.Container._static_scaled_tanh(x, alpha=2, beta=2.5)
        >>> y
        {
            a: ivy.array([2., -2.]),
            b: ivy.array([1.99999881, 2.])
        }

        >>> x = ivy.Container(a=ivy.array([0.3, -0.3]), b=ivy.array([33.0, -33.0]))
        >>> y = ivy.Container._static_scaled_tanh(x, alpha=1.5, beta=25)
        >>> y
        {
            a: ivy.array([1.49999905, -1.49999905]),
            b: ivy.array([1.5, -1.5])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "scaled_tanh",
            x,
            alpha=alpha,
            beta=beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def scaled_tanh(
        self: ivy.Container,
        /,
        *,
        alpha: Union[float, ivy.Container] = 1.7159,
        beta: Union[float, ivy.Container] = 0.67,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.scaled_tanh. This
        method simplywraps the function, and so the docstring for
        ivy.scaled_tanh also applies to this method with minimal changes.

        Parameters
        ----------
        x
           input container.
        alpha
           The scaling parameter for the output.
           Determines the amplitude of the tanh function.
           Default: 1.7159
        beta
            The scaling parameter for the input.
            Determines the slope of the tanh function.
            Default: 0.67
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
             a container with the scaled_tanh function applied.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2., 3.]), b=ivy.array([1., 2.]))
        >>> x.scaled_tanh()
        {
            a: ivy.array([1.49570239, 1.65537548]),
            b: ivy.array([1.00376701, 1.49570239])
        }

        >>> x = ivy.Container(a=ivy.array([1., 1.]), b=ivy.array([1., 1.]))
        >>> x.scaled_tanh(alpha=30)
        {
            a: ivy.array([17.54939651, 17.54939651]),
            b: ivy.array([17.54939651, 17.54939651])
        }

        >>> x = ivy.Container(a=ivy.array([20., 21.]), b=ivy.array([3., 1.]))
        >>> x.scaled_tanh(alpha=0.1, beta=-0.4)
        {
            a: ivy.array([-0.09999998, -0.09999999]),
            b: ivy.array([-0.08336546, -0.0379949])
        }
        """
        return self._static_scaled_tanh(
            self,
            alpha=alpha,
            beta=beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_hardshrink(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        lambd: ivy.Container = 0.5,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = False,
        prune_unapplied: Union[bool, ivy.Container] = True,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.hardshrink. This method
        simply wraps the function, and so the docstring for ivy.hardshrink also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        lambd
            Lambda value for hard shrinkage calculation.
        key_chains
            The key-chains to apply or not apply the method to.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
        map_sequences
            Whether to also map method to sequences (lists, tuples).

        Returns
        -------
        ret
            Container with hard shrinkage applied to the leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., -2.]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container._static_hardshrink(x)
        >>> print(y)
        {
            a: ivy.array([1., -2.]),
            b: ivy.array([0., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hardshrink",
            x,
            lambd=lambd,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hardshrink(
        self: ivy.Container,
        /,
        *,
        lambd: ivy.Container = 0.5,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = False,
        prune_unapplied: Union[bool, ivy.Container] = True,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Apply the hard shrinkage function element-wise.

        Parameters
        ----------
        self
            Input container.
        lambd
            Lambda value for hard shrinkage calculation.
        key_chains
            The key-chains to apply or not apply the method to.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container with hard shrinkage applied to the leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., -2.]), b=ivy.array([0.4, -0.2]))
        >>> y = ivy.Container.hardshrink(x)
        >>> print(y)
        {
            a: ivy.array([1., -2.]),
            b: ivy.array([0., 0.])
        }
        """
        return self._static_hardshrink(
            self,
            lambd=lambd,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_hardsilu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method which acts as a wrapper for
        ivy.hardsilu.

        Parameters
        ----------
        x
            input container
        key_chains
            The keychains to apply or not apply the method to. Default is ``None``.
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
        a container containing the output of the hardsilu/hardswish function applied
        to each element in ``x``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.5, -1, 0]), b=ivy.array([0.5, 1., 2]))
        >>> y = ivy.Container._static_hardsilu(x)
        >>> print(y)
        {
            a: ivy.array([-0.20833333, -0.33333334, 0.]),
            b: ivy.array([0.29166666, 0.66666669, 1.66666663])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hardsilu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hardsilu(
        self,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method which acts as a wrapper for
        ivy.hardsilu.

        Parameters
        ----------
        self
            input container
        key_chains
            The keychains to apply or not apply the method to. Default is ``None``.
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
        a container containing the output of the hardsilu/hardswish function applied
        to each element in the input container.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.5, -1, 0]), b=ivy.array([0.5, 1., 2]))
        >>> y = x.hardsilu()
        >>> print(y)
        {
            a: ivy.array([-0.20833333, -0.33333334, 0.]),
            b: ivy.array([0.29166666, 0.66666669, 1.66666663])
        }
        """
        return self._static_hardsilu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
