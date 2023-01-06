# global
from typing import Union, Optional

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithActivationExperimental(ContainerBase):
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
