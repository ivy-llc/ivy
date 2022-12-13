# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithRandomExperimental(ContainerBase):
    # dirichlet
    @staticmethod
    def static_dirichlet(
        alpha: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        size: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.dirichlet. This method
        simply wraps the function, and so the docstring for ivy.dirichlet also
        applies to this method with minimal changes.

        Parameters
        ----------
        alpha
            Sequence of floats of length k 
        size
            optional container including ints or tuple of ints, 
            Output shape for the arrays in the input container. 
        dtype
            output container array data type. If ``dtype`` is ``None``, the output data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the drawn samples.

        Examples
        --------
        >>> alpha = ivy.Container(a=ivy.array([7,6,5]), \
                                  b=ivy.array([8,9,4]))
        >>> size = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_dirichlet(alpha, size)
        {
            a: ivy.array(
                [[0.43643127, 0.32325703, 0.24031169],
                 [0.34251311, 0.31692529, 0.3405616 ],
                 [0.5319725 , 0.22458365, 0.24344385]]
                ),
            b: ivy.array(
                [[0.26588406, 0.61075421, 0.12336174],
                 [0.51142915, 0.25041268, 0.23815817],
                 [0.64042903, 0.25763214, 0.10193883],
                 [0.31624692, 0.46567987, 0.21807321],
                 [0.37677699, 0.39914594, 0.22407707]]
                )
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "dirichlet",
            alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            size=size,
            dtype=dtype,
            out=out,
        )

    def dirichlet(
        self: ivy.Container,
        /,
        *,
        size: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.dirichlet. This method
        simply wraps the function, and so the docstring for ivy.shuffle also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Sequence of floats of length k 
        size
            optional container including ints or tuple of ints, 
            Output shape for the arrays in the input container. 
        dtype
            output container array data type. If ``dtype`` is ``None``, the output data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the drawn samples.

        Examples
        --------
        >>> alpha = ivy.Container(a=ivy.array([7,6,5]), \
                                  b=ivy.array([8,9,4]))
        >>> size = ivy.Container(a=3, b=5)
        >>> alpha.dirichlet(size)
        {
            a: ivy.array(
                [[0.43643127, 0.32325703, 0.24031169],
                 [0.34251311, 0.31692529, 0.3405616 ],
                 [0.5319725 , 0.22458365, 0.24344385]]
                ),
            b: ivy.array(
                [[0.26588406, 0.61075421, 0.12336174],
                 [0.51142915, 0.25041268, 0.23815817],
                 [0.64042903, 0.25763214, 0.10193883],
                 [0.31624692, 0.46567987, 0.21807321],
                 [0.37677699, 0.39914594, 0.22407707]]
                )
        }
        """
        return self.static_dirichlet(
            self,
            size=size,
            dtype=dtype,
            out=out,
        )
