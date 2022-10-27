# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithElementWiseExtensions(ContainerBase):
    @staticmethod
    def static_sinc(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
        ...                   b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = ivy.Container.static_sinc(x)
        >>> print(y)
        {
            a: ivy.array([0.636, -0.212, 0.127]),
            b: ivy.array([-0.090, 0.070, -0.057])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sinc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinc(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
        ...                   b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = x.sinc()
        >>> print(y)
        {
            a: ivy.array([0.637,-0.212,0.127]),
            b: ivy.array([-0.0909,0.0707,-0.0579])
        }
        """
        return self.static_sinc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_lcm(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),
        ...                  b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),
        ...                  b=ivy.array([32, 40, 25, 13]))
        >>> ivy.Container.lcm(x1, x2)
        {
            a: ivy.array([10, 21, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "lcm",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lcm(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),
        ...                  b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),
        ...                  b=ivy.array([32, 40, 25, 13]))
        >>> x1.lcm(x2)
        {
            a: ivy.array([10, 24, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return self.static_lcm(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_fmod(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmod. This method simply wraps
        the function, and so the docstring for ivy.fmod also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> ivy.Container.static_fmod(x1, x2)
        {
            a: ivy.array([ 0,  3,  0])
            b: ivy.array([ nan,  nan,  nan])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "fmod",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmod(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.fmod. This method simply
        wraps the function, and so the docstring for ivy.fmod also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> x1.fmod(x2)
        {
            a: ivy.array([ 0,  3,  0])
            b: ivy.array([ nan,  nan,  nan])
        }
        """
        return self.static_fmod(self, x2, out=out)

    @staticmethod
    def static_fmax(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmax. This method simply wraps
        the function, and so the docstring for ivy.fmax also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> ivy.Container.static_fmax(x1, x2)
        {
            a: ivy.array([ 2.,  5.,  4.])
            b: ivy.array([ 0,  0,  nan])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "fmax",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmax(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.fmax. This method simply
        wraps the function, and so the docstring for ivy.fmax also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> x1.fmax(x2)
        {
            a: ivy.array([ 2.,  5.,  4.])
            b: ivy.array([ 0,  0,  nan])
        }
        """
        return self.static_fmax(self, x2, out=out)

    @staticmethod
    def static_float_power(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.float_power. This method simply wraps
        the function, and so the docstring for ivy.float_power also applies to this
        method with minimal changes.

        Parameters
        ----------
        x1
            container with the base input arrays.
        x2
            container with the exponent input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with base arrays raised to the powers
            of exponents arrays, element-wise .

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([2, 10]))
        >>> x2 = ivy.Container(a=ivy.array([1, 3, 1]), b=0)
        >>> ivy.Container.static_float_power(x1, x2)
        {
            a: ivy.array([1,  8,  3])
            b: ivy.array([1, 1])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "float_power",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def float_power(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.float_power. This method simply
        wraps the function, and so the docstring for ivy.float_power also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        x2
            container with the exponent input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with base arrays raised to the powers
            of exponents arrays, element-wise .

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([2, 10]))
        >>> x2 = ivy.Container(a=ivy.array([1, 3, 1]), b=0)
        >>> x1.float_power(x2)
        {
            a: ivy.array([1,  8,  3])
            b: ivy.array([1, 1])
        }
        """
        return self.static_float_power(self, x2, out=out)

    @staticmethod
    def static_exp2(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.exp2. This method simply wraps
        the function, and so the docstring for ivy.exp2 also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise 2 to the power
            of input arrays elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=[5, 6, 7])
        >>> ivy.Container.static_exp2(x)
        {
            a: ivy.array([2.,  4.,  8.])
            b: ivy.array([32., 64., 128.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "exp2",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def exp2(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.exp2. This method simply
        wraps the function, and so the docstring for ivy.exp2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise 2 to the power
            of input array elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=[5, 6, 7])
        >>> x.exp2()
        {
            a: ivy.array([2.,  4.,  8.])
            b: ivy.array([32., 64., 128.])
        }
        """
        return self.static_exp2(self, out=out)
