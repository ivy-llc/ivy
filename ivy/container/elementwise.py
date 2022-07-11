# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithElementwise(ContainerBase):
    @staticmethod
    def static_abs(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),\
                            b=ivy.array([4.5, -5.3, -0, -2.3]))
        >>> y = ivy.Container.static_abs(x)
        >>> print(y)
        {
            a: ivy.array([0., 2.6, 3.5]),
            b: ivy.array([4.5, 5.3, 0, 2.3])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "abs",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
    
    def abs(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.

        Examples
        --------
        Using :code:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([-1.6, 2.6, -3.5]),\
                            b=ivy.array([4.5, -5.3, -2.3]))
        >>> y = x.abs()
        >>> print(y)
        {
            a: ivy.array([1.6, 2.6, 3.5]),
            b: ivy.array([4.5, 5.3, 2.3])
        }

        """
        return self.static_abs(
            self, 
            key_chains, 
            to_apply, 
            prune_unapplied, 
            map_sequences, 
            out=out
        )

    @staticmethod
    def static_acosh(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "acosh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def acosh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_acosh(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_acos(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "acos",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def acos(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_acos(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_add(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.add. This method simply wraps the
        function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.array([[1.1, 2.3, -3.6]])
        >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\
                            b=ivy.array([[5.], [6.], [7.]]))
        >>> z = ivy.Container.static_add(x, y)
        >>> print(z)
        {
            a: ivy.array([[5.1, 6.3, 0.4],
                          [6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4]]),
            b: ivy.array([[6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4],
                          [8.1, 9.3, 3.4]])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), \
                            b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\
                            b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_add(x, y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 9]),
            b: ivy.array([7, 9, 11])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "add",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def add(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.add. This method simply wraps the
        function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                             b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\
                             b=ivy.array([5, 6, 7]))

        >>> z = x.add(y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 9]),
            b: ivy.array([7, 9, 11])
        }
        """
        return self.static_add(
            self, x2, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_asin(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "asin",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def asin(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_asin(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_asinh(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "asinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_asinh(
        x: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.asinh. This method simply wraps the
        function, and so the docstring for ivy.asinh also applies to this method
        with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([1.5, 0., -3.5]),\
                            b=ivy.array([3.4, -5.3, -0, -2.8]))
        >>> y = ivy.Container.static_asinh(x)
        >>> print(y)
        {
            a: ivy.array([1.19, 0., -1.97]),
            b: ivy.array([1.94, -2.37, 0., -1.75])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "asinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def asinh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.asinh. This method simply wraps the
        function, and so the docstring for ivy.asinh also applies to this method
        with minimal changes.
        Examples
        --------
        Using :code:`ivy.Container` instance method:
        >>> x = ivy.Container(a=ivy.array([-1, 3.7, -5.1]),\
                            b=ivy.array([4.5, -2.4, -1.5]))
        >>> y = x.asinh()
        >>> print(y)
        {
            a: ivy.array([-0.881, 2.02, -2.33]),
            b: ivy.array([2.21, -1.61, -1.19])
        }
        """
        return self.static_asinh(
            self, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_atan(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "atan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )


    def atan(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_atan(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_atan2(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "atan2",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atan2(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_atan2(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_atanh(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "atanh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atanh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_atanh(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_bitwise_and(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "bitwise_and",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_and(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_bitwise_and(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_bitwise_left_shift(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "bitwise_left_shift",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_left_shift(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_bitwise_left_shift(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_bitwise_invert(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "bitwise_invert",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_invert(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_bitwise_invert(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_bitwise_or(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "bitwise_or",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_or(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_bitwise_or(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_bitwise_right_shift(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "bitwise_right_shift",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_right_shift(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_bitwise_right_shift(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_bitwise_xor(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "bitwise_xor",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_xor(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_bitwise_xor(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_ceil(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "ceil",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ceil(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.ceil. This method simply wraps the
        function, and so the docstring for ivy.ceil also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.5, 0.5, -1.4]),\
                              b=ivy.array([5.4, -3.2, 5.2]))
        >>> y = x.ceil()
        >>> print(y)
        {
            a: ivy.array([3., 1., -1.]),
            b: ivy.array([6., -3., 6.])
        }
        """
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.ceil(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    @staticmethod
    def static_cos(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "cos",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cos(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_cos(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_cosh(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "cosh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cosh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_cosh(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_divide(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def divide(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_divide(
            self, x2, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_equal(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_equal(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_exp(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "exp",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def exp(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_exp(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_expm1(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "expm1",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def expm1(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_expm1(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_floor(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "floor",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def floor(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.floor. This method simply wraps the
        function, and so the docstring for ivy.floor also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.5, 0.5, -1.4]),\
                              b=ivy.array([5.4, -3.2, 5.2]))
        >>> y = x.floor()
        >>> print(y)
        {
            a: ivy.array([2., 0., -2.]),
            b: ivy.array([5., -4., 5.])
        }
        """
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.floor(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    @staticmethod
    def static_floor_divide(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "floor_divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def floor_divide(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_floor_divide(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_greater(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "greater",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def greater(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_greater(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_greater_equal(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "greater_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def greater_equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_greater_equal(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_isfinite(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "isfinite",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isfinite(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_isfinite(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_isinf(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "isinf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isinf(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_isinf(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_isnan(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "isnan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isnan(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_isnan(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_less(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "less",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def less(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_less(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_less_equal(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "less_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def less_equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_atan2(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_log(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "log",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_log(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_log1p(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "log1p",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log1p(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_log1p(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_log2(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "log2",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log2(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_log2(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_log10(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "log10",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log10(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_log10(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_logaddexp(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "logaddexp",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logaddexp(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_logaddexp(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_logical_and(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "logical_and",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_and(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_logical_and(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_logical_not(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "logical_not",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_not(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_logical_not(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_logical_or(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "logical_or",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_or(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_logical_or(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_logical_xor(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "logical_xor",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_xor(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_logical_xor(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_multiply(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "multiply",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def multiply(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_multiply(
            self, x2, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_negative(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "negative",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def negative(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_negative(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_not_equal(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "not_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def not_equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_not_equal(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_positive(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "positive",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def positive(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_positive(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_pow(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "pow",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def pow(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_nests: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"x1": self}
        if ivy.is_array(x2):
            kw["x2"] = x2
        else:
            conts["x2"] = x2
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.pow(**dict(zip(conts.keys(), xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
            out=out,
        )

    @staticmethod
    def static_remainder(
            x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "remainder",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def remainder(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_remainder(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_round(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "round",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def round(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_round(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_sign(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "sign",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sign(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_sign(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_sin(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "sin",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sin(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_sin(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_sinh(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "sinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_sinh(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_square(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "square",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def square(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_square(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_sqrt(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "sqrt",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sqrt(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_sqrt(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_subtract(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "subtract",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def subtract(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_subtract(
            self, x2, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_tan(
        x: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.tan. This method simply wraps the
        function, and so the docstring for ivy.tan also applies to this method
        with minimal changes.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_tan(x)
        >>> print(y)
        {
            a: ivy.array([0., 1.56, -2.19]),
            b: ivy.array([-0.143, 1.16, -3.38])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "tan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tan(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.tan. This method simply wraps the
        function, and so the docstring for ivy.tan also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.tan()
        >>> print(y)
        {
            a:ivy.array([0., 1.56, -2.19]),
            b:ivy.array([-0.143, 1.16, -3.38])
        }
        """
        return self.static_tan(
            self, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_tanh(
            x: ivy.Container,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.tanh. This method simply wraps the
        function, and so the docstring for ivy.tanh also applies to this method
        with minimal changes.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_tanh(x)
        >>> print(y)
        {
            a: ivy.array([0., 0.76, 0.96]),
            b: ivy.array([0.995, 0.999, 0.9999])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "tanh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tanh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.tanh. This method simply wraps the
        function, and so the docstring for ivy.tanh also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),\
                              b=ivy.array([3., 4., 5.]))
        >>> y = x.tanh()
        >>> print(y)
        {
            a:ivy.array([0., 0.762, 0.964]),
            b:ivy.array([0.995, 0.999, 1.])
        }
        """
        return self.static_tanh(
            self, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )

    @staticmethod
    def static_trunc(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "trunc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trunc(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_trunc(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )

    @staticmethod
    def static_erf(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "erf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def erf(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_erf(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out
        )
