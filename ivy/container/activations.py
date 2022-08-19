# local
import ivy
from ivy.container.base import ContainerBase
from typing import Optional, Union, List, Dict


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithActivations(ContainerBase):
    @staticmethod
    def static_relu(
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
        ivy.Container static method variant of ivy.relu.
        This method simply wraps the function, and so the docstring
        for ivy.relu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        >>> x = ivy.Container(a=ivy.array([1.0, 0, 1.0]))
        >>> y = ivy.Container.static_relu(x)
        >>> print(y)
        {
            a: ivy.array([1., 0., 1.])
        }

        """
        return ContainerBase.multi_map_in_static_method(
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
        ivy.Container instance method variant of ivy.relu.
        This method simply wraps the function, and so the docstring
        for ivy.relu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        >>> x = ivy.Container(a=ivy.array([1.0, 0, 1.0]))
        >>> y = x.relu()
        >>> print(y)
        {
            a: ivy.array([1., 0., 1.])
        }

        """
        return self.static_relu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_leaky_relu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        alpha: Optional[ivy.Container] = 0.2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.leaky_relu.
        This method simply wraps the function, and so the docstring
        for ivy.leaky_relu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        alpha
            array or scalar specifying the negative slope.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
             a container with the leaky relu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a =ivy.array([0.39, -0.85]))
        >>> y = ivy.Container.static_leaky_relu(x)
        >>> print(y)
        {
              a: ivy.array([0.39, -0.17])
        }

        """
        return ContainerBase.multi_map_in_static_method(
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
        alpha: Optional[ivy.Container] = 0.2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.leaky_relu.
        This method simply wraps the function, and so the docstring
        for ivy.leaky_relu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        alpha
            array or scalar specifying the negative slope.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
           a container with the leaky relu unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a =ivy.array([0.39, -0.85]))
        >>> y = x.leaky_relu()
        >>> print(y)
        {
            a: ivy.array([0.39, -0.17])
        }

        """
        return self.static_leaky_relu(
            self,
            alpha=alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_gelu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        approximate: Optional[ivy.Container] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.gelu.
        This method simply wraps the function, and so the docstring
        for ivy.gelu also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        approximate
            whether to use the gelu approximation algorithm or exact formulation.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        return ContainerBase.multi_map_in_static_method(
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
        approximate: Optional[bool] = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gelu.
        This method simply wraps the function, and so the docstring
        for ivy.gelu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        approximate
            whether to use the gelu approximation algorithm or exact formulation.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        >>> y = x.gelu()
        >>> print(y)
        {
            a: ivy.array([0.185, -0.046])
        }

        """
        return self.static_gelu(
            self,
            approximate=approximate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sigmoid(
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
        ivy.Container static method variant of ivy.sigmoid.
        This method simply wraps the function, and so the docstring
        for ivy.sigmoid also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the sigmoid unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1., 1., 2.]))
        >>> y = ivy.Container.static_sigmoid(x)
        >>> print(y)
        {
            a: ivy.array([0.269, 0.731, 0.881])
        }

        """
        return ContainerBase.multi_map_in_static_method(
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
        ivy.Container instance method variant of ivy.sigmoid.
        This method simply wraps the function, and so the docstring
        for ivy.sigmoid also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the sigmoid unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1., 1., 2.]))
        >>> y = x.sigmoid()
        >>> print(y)
        {
            a: ivy.array([0.269, 0.731, 0.881])
        }

        """
        return self.static_sigmoid(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_softmax(
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
        ivy.Container static method variant of ivy.softmax.
        This method simply wraps the function, and so the docstring
        for ivy.softmax also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            the axis or axes along which the softmax should be computed
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, 0, 1.0]))
        >>> y = ivy.Container.static_softmax(x)
        >>> print(y)
        {
            a: ivy.array([0.422, 0.155, 0.422])
        }

        """
        return ContainerBase.multi_map_in_static_method(
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
        ivy.Container instance method variant of ivy.softmax.
        This method simply wraps the function, and so the docstring
        for ivy.softmax also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            the axis or axes along which the softmax should be computed
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.0, 0, 1.0]))
        >>> y = x.softmax()
        >>> print(y)
        {
            a: ivy.array([0.422, 0.155, 0.422])
        }

        """
        return self.static_softmax(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_softplus(
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
        ivy.Container static method variant of ivy.softplus.
        This method simply wraps the function, and so the docstring
        for ivy.softplus also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        >>> y = ivy.Container.static_softplus(x)
        >>> print(y)
        {
            a: ivy.array([0.535, 0.42])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "softplus",
            x,
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
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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

        """
        return self.static_softplus(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
