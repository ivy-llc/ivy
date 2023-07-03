# local
import ivy
from ivy.data_classes.container.base import ContainerBase
from typing import Optional, Union, List, Dict


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class _ContainerWithActivations(ContainerBase):
    @staticmethod
    def _static_relu(
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
        ivy.Container static method variant of ivy.relu. This method simply wraps the
        function, and so the docstring for ivy.relu also applies to this method with
        minimal changes.

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
        >>> y = ivy.Container.static_relu(x)
        >>> print(y)
        {
            a: ivy.array([1., 0.]),
            b: ivy.array([0.40000001, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "relu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def relu(
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
        ivy.Container instance method variant of ivy.relu. This method simply wraps the
        function, and so the docstring for ivy.relu also applies to this method with
        minimal changes.

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
        >>> y = x.relu()
        >>> print(y)
        {
            a: ivy.array([1., 0.]),
            b: ivy.array([0.40000001, 0.])
        }
        """
        return self._static_relu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_leaky_relu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: ivy.Container = 0.2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.leaky_relu. This method simply wraps
        the function, and so the docstring for ivy.leaky_relu also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input container.
        alpha
            array or scalar specifying the negative slope.
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
             a container with the leaky relu unit function applied element-wise.

        Examples
        --------
        >>> x = x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = ivy.Container.static_leaky_relu(x)
        >>> print(y)
        {
            a: ivy.array([0.38999999, -0.17]),
            b: ivy.array([1., -0.04])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "leaky_relu",
            x,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def leaky_relu(
        self: ivy.Container,
        /,
        *,
        alpha: ivy.Container = 0.2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.leaky_relu. This method simply
        wraps the function, and so the docstring for ivy.leaky_relu also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input container.
        alpha
            array or scalar specifying the negative slope.
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
           a container with the leaky relu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
        >>> y = x.leaky_relu()
        >>> print(y)
        {
            a: ivy.array([0.38999999, -0.17]),
            b: ivy.array([1., -0.04])
        }
        """
        return self._static_leaky_relu(
            self,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_gelu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        approximate: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.gelu. This method simply wraps the
        function, and so the docstring for ivy.gelu also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container.
        approximate
            whether to use the gelu approximation algorithm or exact formulation.
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
            a container with the gelu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a =ivy.array([0.3, -0.1]))
        >>> y = ivy.Container.static_gelu(x)
        >>> print(y)
        {
             a: ivy.array([0.185, -0.046])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "gelu",
            x,
            approximate=approximate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gelu(
        self: ivy.Container,
        /,
        *,
        approximate: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gelu. This method simply wraps the
        function, and so the docstring for ivy.gelu also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container.
        approximate
            whether to use the gelu approximation algorithm or exact formulation.
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
            a container with the gelu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([-0.9, -1.]))
        >>> y = x.gelu()
            print(y)
            {
                 a: ivy.array([0.841, 1.95]),
                 b: ivy.array([-0.166, -0.159])
            }
        """
        return self._static_gelu(
            self,
            approximate=approximate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sigmoid(
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
        ivy.Container static method variant of ivy.sigmoid. This method simply wraps the
        function, and so the docstring for ivy.sigmoid also applies to this method with
        minimal changes.

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
            a container with the sigmoid unit function applied element-wise.

        Examples
        --------
        >>> ivy.Container(a=ivy.array([-1., 1., 2.]), b=ivy.array([0.5, 0., -0.1]))
        >>> y = ivy.Container.static_sigmoid(x)
        >>> print(y)
        {
            a: ivy.array([0.2689414, 0.7310586, 0.88079703]),
            b: ivy.array([0.62245935, 0.5, 0.4750208])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sigmoid",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sigmoid(
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
        ivy.Container instance method variant of ivy.sigmoid. This method simply wraps
        the function, and so the docstring for ivy.sigmoid also applies to this method
        with minimal changes.

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
        >>> y = x.sigmoid()
        >>> print(y)
        {
            a: ivy.array([0.2689414, 0.7310586, 0.88079703]),
            b: ivy.array([0.62245935, 0.5, 0.4750208])
        }
        """
        return self._static_sigmoid(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_softmax(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.softmax. This method simply wraps the
        function, and so the docstring for ivy.softmax also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            the axis or axes along which the softmax should be computed
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
            a container with the softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, 0]), b=ivy.array([1.3, 0, -1.0]))
        >>> y = ivy.Container.static_softmax(x)
        >>> print(y)
        {
            a: ivy.array([0.7310586, 0.2689414]),
            b: ivy.array([0.72844321, 0.19852395, 0.07303288])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "softmax",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def softmax(
        self: ivy.Container,
        /,
        *,
        axis: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softmax. This method simply wraps
        the function, and so the docstring for ivy.softmax also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            the axis or axes along which the softmax should be computed
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
            a container with the softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, 0]), b=ivy.array([1.3, 0, -1.0]))
        >>> y = x.softmax()
        >>> print(y)
        {
            a: ivy.array([0.7310586, 0.2689414]),
            b: ivy.array([0.72844321, 0.19852395, 0.07303288])
        }
        """
        return self._static_softmax(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_softplus(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        beta: Optional[Union[int, float]] = None,
        threshold: Optional[Union[int, float]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.softplus. This method simply wraps
        the function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        beta
            The beta value for the softplus formation. Default: ``None``.
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the softplus unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]), b=ivy.array([1., 0.]))
        >>> y = ivy.Container.static_softplus(x)
        >>> print(y)
        {
            a: ivy.array([0.53499615, 0.42036411]),
            b: ivy.array([1.31326175, 0.69314718])
        }

        >>> x = ivy.Container(a=ivy.array([-1., 2., 4.]))
        >>> y = ivy.Container.static_softplus(x, beta=0.5, threshold=2)
        >>> print(y)
        {
            a: ivy.array([0.948, 2.63, 4.25])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "softplus",
            x,
            beta=beta,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def softplus(
        self: ivy.Container,
        /,
        *,
        beta: Optional[Union[int, float]] = None,
        threshold: Optional[Union[int, float]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softplus. This method simply wraps
        the function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        beta
            The beta value for the softplus formation. Default: ``None``.
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the softplus unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-0.3461, -0.6491]))
        >>> y = x.softplus()
        >>> print(y)
        {
            a: ivy.array([0.535, 0.42])
        }

        >>> x = ivy.Container(a=ivy.array([-1., 2., 4.]))
        >>> y = x.softplus(beta=0.5, threshold=2)
        >>> print(y)
        {
            a: ivy.array([0.948, 2.63, 4.25])
        }
        """
        return self._static_softplus(
            self,
            beta=beta,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_log_softmax(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.log_softmax. This method simply wraps
        the function, and so the docstring for ivy.log_softmax also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            the axis or axes along which the log_softmax should be computed
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
            a container with the log_softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1.0, -0.98, 2.3]))
        >>> y = ivy.Container.static_log_softmax(x)
        >>> print(y)
        {
            a: ivy.array([-3.37, -3.35, -0.0719])
        }

        >>> x = ivy.Container(a=ivy.array([1.0, 2.4]), b=ivy.array([-0.2, -1.0]))
        >>> y = ivy.Container.static_log_softmax(x)
        >>> print(y)
        {
            a: ivy.array([-1.62, -0.22]),
            b: ivy.array([-0.371, -1.17])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "log_softmax",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log_softmax(
        self: ivy.Container,
        /,
        *,
        axis: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container instance method variant of ivy.log_softmax. This method simply
        wraps the function, and so the docstring for ivy.log_softmax also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            the axis or axes along which the log_softmax should be computed
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
            a container with the log_softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1.0, -0.98, 2.3]))
        >>> y = x.log_softmax()
        >>> print(y)
        {
            a: ivy.array([-3.37, -3.35, -0.0719])
        }

        >>> x = ivy.Container(a=ivy.array([1.0, 2.4]), b=ivy.array([-0.2, -1.0]))
        >>> y = x.log_softmax()
        >>> print(y)
        {
            a: ivy.array([-1.62, -0.22]),
            b: ivy.array([-0.371, -1.17])
        }
        """
        return self._static_log_softmax(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_mish(
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
        ivy.Container static method variant of ivy.mish. This method simply wraps the
        function, and so the docstring for ivy.mish also applies to this method with
        minimal changes.

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
        >>> y = ivy.Container.static_mish(x)
        >>> print(y)
        {
            a: ivy.array([0.86509842, -0.30883577]),
            b: ivy.array([0.28903052, -0.10714479])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "mish",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def mish(
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
        ivy.Container instance method variant of ivy.mish. This method simply wraps the
        function, and so the docstring for ivy.mish also applies to this method with
        minimal changes.

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
        >>> y = x.mish()
        >>> print(y)
        {
            a: ivy.array([0.86509842, -0.30883577]),
            b: ivy.array([0.28903052, -0.10714479])
        }
        """
        return self._static_mish(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_hardswish(
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
        ivy.Container static method variant of ivy.hardswish. This method simply wraps
        the function, and so the docstring for ivy.hardswish also applies to this method
        with minimal changes.

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
            a container with the hardswish activation function applied
            element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-3., 4., 5.]), b=ivy.array([0., 5.]))
        >>> x = ivy.hardswish(x, out=x)
        >>> x
        {
            a: ivy.array([-0.,  4.,  5.]),
            b: ivy.array([0., 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hardswish",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hardswish(
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
        ivy.Container instance method variant of ivy.hardswish. This method simply wraps
        the function, and so the docstring for ivy.hardswish also applies to this method
        with minimal changes.

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
            a container with the hardswish activation function applied
            element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-3., 4., 5.]), b=ivy.array([0., 5.]))
        >>> x = ivy.hardswish(x, out=x)
        >>> x
        {
            a: ivy.array([-0.,  4.,  5.]),
            b: ivy.array([0., 5.])
        }
        """
        return self._static_hardswish(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
