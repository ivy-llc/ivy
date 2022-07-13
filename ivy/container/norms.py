# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithNorms(ContainerBase):
    def static_layer_norm(
        x: Union[float, ivy.Container, ivy.Array, ivy.NativeArray],
        normalized_idxs: List[int],
        epsilon: float = ivy._MIN_BASE,
        scale: float = 1.0,
        offset: float = 1.0,
        new_std: float = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.layer_norm. This method simply wraps
        the function, and so the docstring for ivy.layer_norm also applies to this
        method with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                               'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = [0]
        >>> norm = x.static_layer_norm(x, normalized_idxs, 1.25, 0.3)
        >>> print(norm)
        {
            a: ivy.array([0.658, 1.04, 1.3]),
            b: ivy.array([[0.759, 0.759, 0.759], 
                          [1.24, 1.24, 1.24]])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                               'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
        >>> new_std = ivy.Container({'a': 1.25, 'b': 1.5})
        >>> offset = ivy.Container({'a': 0.2, 'b': 0.3})
        >>> norm = x.static_layer_norm(x, normalized_idxs, \
                                       new_std=new_std, offset=offset)
        >>> print(norm)
        {
            a: ivy.array([-1.42, 0.403, 1.62]),
            b: ivy.array([[-1.54, 0.3, 2.14], 
                          [-1.54, 0.3, 2.14]])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "layer_norm",
            x,
            normalized_idxs,
            epsilon,
            scale,
            offset,
            new_std,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def layer_norm(
        self: ivy.Container,
        normalized_idxs: List[int],
        epsilon: float = ivy._MIN_BASE,
        scale: float = 1.0,
        offset: float = 1.0,
        new_std: float = 1.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.layer_norm. This method simply
        wraps the function, and so the docstring for ivy.layer_norm also applies
        to this method with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                               'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = [0]
        >>> norm = x.layer_norm(normalized_idxs, 1.25, 0.3)
        >>> print(norm)
        {
            a: ivy.array([0.658, 1.04, 1.3]),
            b: ivy.array([[0.759, 0.759, 0.759], 
                          [1.24, 1.24, 1.24]])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                               'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
        >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
        >>> new_std = ivy.Container({'a': 1.25, 'b': 1.5})
        >>> offset = ivy.Container({'a': 0.2, 'b': 0.3})
        >>> norm = x.layer_norm(normalized_idxs, new_std=new_std, offset=offset)
        >>> print(norm)
        {
            a: ivy.array([-1.42, 0.403, 1.62]),
            b: ivy.array([[-1.54, 0.3, 2.14], 
                          [-1.54, 0.3, 2.14]])
        }

        """
        return self.static_layer_norm(
            normalized_idxs,
            epsilon,
            scale,
            offset,
            new_std,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
