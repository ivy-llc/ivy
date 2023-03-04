# global
from typing import Union, Optional, List, Dict, Tuple, Sequence

# local
from ivy.container.base import ContainerBase
import ivy


class _ContainerWithLinearAlgebraExperimental(ContainerBase):
    @staticmethod
    def static_eigh_tridiagonal(
        alpha: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        beta: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        eigvals_only: bool = True,
        select: str = "a",
        select_range: Optional[
            Union[Tuple[int, int], List[int], ivy.Array, ivy.NativeArray]
        ] = None,
        tol: Optional[float] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Union[ivy.Container, Tuple[ivy.Container, ivy.Container]]:
        """
        ivy.Container static method variant of ivy.eigh_tridiagonal. This method simply
        wraps the function, and so the docstring for ivy.eigh_tridiagonal also applies
        to this method with minimal changes.

        Parameters
        ----------
        alpha
            An array or a container of real or complex arrays each of
            shape (n), the diagonal elements of the matrix.
        beta
            An array or a container of real or complex arrays each of shape (n-1),
            containing the elements of the first super-diagonal of the matrix.
        eigvals_only
            If False, both eigenvalues and corresponding eigenvectors are computed.
            If True, only eigenvalues are computed. Default is True.
        select
            Optional string with values in {'a', 'v', 'i'}
            (default is 'a') that determines which eigenvalues
            to calculate: 'a': all eigenvalues. 'v': eigenvalues
            in the interval (min, max] given by select_range.
            'i': eigenvalues with indices min <= i <= max.
        select_range
            Size 2 tuple or list or array specifying the range of
            eigenvalues to compute together with select. If select
            is 'a', select_range is ignored.
        tol
            Optional scalar. Ignored when backend is not Tensorflow. The
            absolute tolerance to which each eigenvalue is required. An
            eigenvalue (or cluster) is considered to have converged if
            it lies in an interval of this width. If tol is None (default),
            the value eps*|T|_2 is used where eps is the machine precision,
            and |T|_2 is the 2-norm of the matrix T.

        Returns
        -------
        eig_vals
            The eigenvalues of the matrix in non-decreasing order.
        eig_vectors
            If eigvals_only is False the eigenvectors are returned in the second
            output argument.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> alpha = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([2., 2., 2.]))
        >>> beta = ivy.array([0.,2.])
        >>> y = ivy.Container.static_eigh_tridiagonal(alpha, beta)
        >>> print(y)
        {
            a: ivy.array([-0.56155, 0., 3.56155]),
            b: ivy.array([0., 2., 4.])
        }

        >>> alpha = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([2., 2., 2.]))
        >>> beta = ivy.Container(a=ivy.array([0.,2.]), b=ivy.array([2.,2.]))
        >>> y = ivy.Container.static_eigh_tridiagonal(alpha, beta)
        >>> print(y)
        {
            a: ivy.array([-0.56155, 0., 3.56155]),
            b: ivy.array([-0.82842, 2., 4.82842])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "eigh_tridiagonal",
            alpha,
            beta,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def eigh_tridiagonal(
        self: ivy.Container,
        beta: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        eigvals_only: bool = True,
        select: str = "a",
        select_range: Optional[
            Union[Tuple[int, int], List[int], ivy.Array, ivy.NativeArray]
        ] = None,
        tol: Optional[float] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Union[ivy.Container, Tuple[ivy.Container, ivy.Container]]:
        """
        ivy.Container instance method variant of ivy.eigh_tridiagonal.
        This method simply wraps the function, and so the docstring for
        ivy.eigh_tridiagonal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            A container of real or complex arrays each of shape (n),
            the diagonal elements of the matrix.
        beta
            An array or a container of real or complex arrays each of shape
            (n-1), containing the elements of the first super-diagonal of the matrix.
        eigvals_only
            If False, both eigenvalues and corresponding eigenvectors are computed.
            If True, only eigenvalues are computed. Default is True.
        select
            Optional string with values in {'a', 'v', 'i'} (default is 'a') that
            determines which eigenvalues to calculate: 'a': all eigenvalues.
            'v': eigenvalues in the interval (min, max] given by select_range.
            'i': eigenvalues with indices min <= i <= max.
        select_range
            Size 2 tuple or list or array specifying the range of eigenvalues to
            compute together with select. If select is 'a', select_range is ignored.
        tol
            Optional scalar. Ignored when backend is not Tensorflow. The absolute
            tolerance to which each eigenvalue is required. An eigenvalue (or cluster)
            is considered to have converged if it lies in an interval of this width.
            If tol is None (default), the value eps*|T|_2 is used where eps is the
            machine precision, and |T|_2 is the 2-norm of the matrix T.

        Returns
        -------
        eig_vals
            The eigenvalues of the matrix in non-decreasing order.
        eig_vectors
            If eigvals_only is False the eigenvectors are returned in
            the second output argument.

        Examples
        --------
        >>> alpha = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([2., 2., 2.]))
        >>> beta = ivy.array([0.,2.])
        >>> y = alpha.eigh_tridiagonal(beta)
        >>> print(y)
        {
            a: ivy.array([-0.56155, 0., 3.56155]),
            b: ivy.array([0., 2., 4.])
        }

        >>> alpha = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([2., 2., 2.]))
        >>> beta = ivy.Container(a=ivy.array([0.,2.]), b=ivy.array([2.,2.]))
        >>> y = alpha.eigh_tridiagonal(beta)
        >>> print(y)
        {
            a: ivy.array([-0.56155, 0., 3.56155]),
            b: ivy.array([-0.82842, 2., 4.82842])
        }
        """
        return self.static_eigh_tridiagonal(
            self,
            beta,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_diagflat(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        offset: int = 0,
        padding_value: float = 0,
        align: str = "RIGHT_LEFT",
        num_rows: int = -1,
        num_cols: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "diagflat",
            x,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def diagflat(
        self: ivy.Container,
        /,
        *,
        offset: int = 0,
        padding_value: float = 0,
        align: str = "RIGHT_LEFT",
        num_rows: int = -1,
        num_cols: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.diagflat.
        This method simply wraps the function, and so the docstring for
        ivy.diagflat also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=[1,2])
        >>> ivy.diagflat(x, k=1)
        {
            a: ivy.array([[0, 1, 0],
                          [0, 0, 2],
                          [0, 0, 0]])
        }
        """
        return self.static_diagflat(
            self,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_kron(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        b: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.kron. This method simply wraps
        the function, and so the docstring for ivy.kron also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            first container with input arrays.
        b
            second container with input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the Kronecker product of
            the arrays in the input containers, computed element-wise

        Examples
        --------
        >>> a = ivy.Container(x=ivy.array([1,2]), y=ivy.array(50))
        >>> b = ivy.Container(x=ivy.array([3,4]), y=ivy.array(9))
        >>> ivy.Container.static_kron(a, b)
        {
            a: ivy.array([3, 4, 6, 8])
            b: ivy.array([450])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "kron",
            a,
            b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def kron(
        self: ivy.Container,
        b: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.kron.
        This method simply wraps the function, and so the docstring for
        ivy.kron also applies to this method with minimal changes.

        Examples
        --------
        >>> a = ivy.Container(x=ivy.array([1,2]), y=ivy.array([50]))
        >>> b = ivy.Container(x=ivy.array([3,4]), y=ivy.array(9))
        >>> a.kron(b)
        {
            a: ivy.array([3, 4, 6, 8])
            b: ivy.array([450])
        }
        """
        return self.static_kron(
            self,
            b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_matrix_exp(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "matrix_exp",
            x,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
        )

    def matrix_exp(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.diagflat.
        This method simply wraps the function, and so the docstring for
        ivy.diagflat also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([[[1., 0.],
                            [0., 1.]],
                            [[2., 0.],
                            [0., 2.]]])
        >>> ivy.matrix_exp(x)
        ivy.array([[[2.7183, 1.0000],
                    [1.0000, 2.7183]],
                    [[7.3891, 1.0000],
                    [1.0000, 7.3891]]])
        """
        return self.static_matrix_exp(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            out=out,
        )

    @staticmethod
    def static_eig(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.eig.
        This method simply wraps the function, and so the docstring for
        ivy.eig also applies to this method with minimal changes.

        Parameters
        ----------
            x
                container with input arrays.

        Returns
        -------
            ret
                container including tuple of arrays corresponding to
                eigenvealues and eigenvectors of input array

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> c = ivy.Container({'x':{'xx':x}})
        >>> ivy.Container.eig(c)
        {
            x:  {
                    xx: (tuple(2), <class ivy.array.array.Array>, shape=[2, 2])
                }
        }
        >>> ivy.Container.eig(c)['x']['xx']
        (
            ivy.array([-0.37228107+0.j,  5.3722816 +0.j]),
            ivy.array([
                    [-0.8245648 +0.j, -0.41597357+0.j],
                    [0.56576747+0.j, -0.9093767 +0.j]
                ])
        )
        """
        return ContainerBase.cont_multi_map_in_function(
            "eig",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def eig(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.eig.
        This method simply wraps the function, and so the docstring for
        ivy.eig also applies to this method with minimal changes.

        Parameters
        ----------
            x
                container with input arrays.

        Returns
        -------
            ret
                container including arrays corresponding
                eigenvealues and eigenvectors of input arrays

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> c = ivy.Container({'x':{'xx':x}})
        >>> c.eig()
        {
            x:  {
                    xx: (tuple(2), <class ivy.array.array.Array>, shape=[2, 2])
                }
        }
        >>>c.eig()['x']['xx']
        (
            ivy.array([-0.37228107+0.j,  5.3722816 +0.j]),
            ivy.array([
                    [-0.8245648 +0.j, -0.41597357+0.j],
                    [0.56576747+0.j, -0.9093767 +0.j]
                ])
        )
        """
        return self.static_eig(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_eigvals(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.eigvals.
        This method simply wraps the function, and so the docstring for
        ivy.eigvals also applies to this method with minimal changes.

        Parameters
        ----------
            x
                container with input arrays.

        Returns
        -------
            ret
                container including array corresponding
                to eigenvalues of input array

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> c = ivy.Container({'x':{'xx':x}})
        >>> ivy.Container.eigvals(c)
        {
            x: {
                xx: ivy.array([-0.37228132+0.j, 5.37228132+0.j])
            }
        }
        >>> ivy.Container.eigvals(c)['x']['xx']
        ivy.array([-0.37228132+0.j,  5.37228132+0.j])
        """
        return ContainerBase.cont_multi_map_in_function(
            "eigvals",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def eigvals(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.eigvals.
        This method simply wraps the function, and so the docstring for
        ivy.eigvals also applies to this method with minimal changes.

        Parameters
        ----------
            x
                container with input arrays.

        Returns
        -------
            ret
                container including array corresponding
                to eigenvalues of input array

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> c = ivy.Container({'x':{'xx':x}})
        >>> c.eigvals()
        {
            x: {
                xx: ivy.array([-0.37228132+0.j, 5.37228132+0.j])
            }
        }
        >>> c.eigvals()['x']['xx']
        ivy.array([-0.37228132+0.j,  5.37228132+0.j])
        """
        return self.static_eigvals(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_adjoint(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container static method variant of ivy.adjoint. This method simply wraps
        the function, and so the docstring for ivy.adjoint also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            container with input arrays of dimensions greater than 1.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the conjugate transpose of
            the arrays in the input container

        Examples
        --------
        >>> x = np.array([[1.-1.j, 2.+2.j],
                          [3.+3.j, 4.-4.j]])
        >>> y = np.array([[1.-2.j, 3.+4.j],
                          [1.-0.j, 2.+6.j]])
        >>> c = ivy.Container(a=ivy.array(x), b=ivy.array(y))
        >>> ivy.Container.static_adjoint(c)
        {
            a: ivy.array([[1.+1.j, 3.-3.j],
                          [2.-2.j, 4.+4.j]]),
            b: ivy.array([[1.+2.j, 1.-0.j],
                          [3.-4.j, 2.-6.j]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "adjoint",
            x,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
        )

    def adjoint(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container instance method variant of ivy.adjoint.
        This method simply wraps the function, and so the docstring for
        ivy.adjoint also applies to this method with minimal changes.

        Examples
        --------
        >>> x = np.array([[1.-1.j, 2.+2.j],
                          [3.+3.j, 4.-4.j]])
        >>> c = ivy.Container(a=ivy.array(x))
        >>> c.adjoint()
        {
            a: ivy.array([[1.+1.j, 3.-3.j],
                          [2.-2.j, 4.+4.j]])
        }
        """
        return self.static_adjoint(
            self, key_chains=key_chains, to_apply=to_apply, out=out
        )

    @staticmethod
    def static_multi_dot(
        x: Sequence[Union[ivy.Container, ivy.Array, ivy.NativeArray]],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.multi_dot. This method simply wraps
        the function, and so the docstring for ivy.multi_dot also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            sequence of matrices to multiply.
        out
            optional output array, for writing the result to. It must have a valid
            shape, i.e. the resulting shape after applying regular matrix multiplication
            to the inputs.

        Returns
        -------
        ret
            dot product of the arrays.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> a = ivy.Container(x=ivy.arange(2 * 3).reshape((2, 3)),
        ...                   y=ivy.arange(2 * 3).reshape((2, 3)))
        >>> b = ivy.Container(x=ivy.arange(3 * 2).reshape((3, 2)),
        ...                   y=ivy.arange(3 * 2).reshape((3, 2)))
        >>> c = ivy.Container(x=ivy.arange(2 * 2).reshape((2, 2)),
        ...                   y=ivy.arange(2 * 2).reshape((2, 2)))
        >>> ivy.Container.static_multi_dot((a, b, c))
        {
            x: ivy.array([[26, 49],
                          [80, 148]]),
            y: ivy.array([[26, 49],
                          [80, 148]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "multi_dot",
            x,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def multi_dot(
        self: ivy.Container,
        arrays: Sequence[Union[ivy.Container, ivy.Array, ivy.NativeArray]],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.multi_dot.
        This method simply wraps the function, and so the docstring for
        ivy.multi_dot also applies to this method with minimal changes.

        Examples
        --------
        >>> a = ivy.Container(x=ivy.arange(2 * 3).reshape((2, 3)),
        ...                   y=ivy.arange(2 * 3).reshape((2, 3)))
        >>> b = ivy.Container(x=ivy.arange(3 * 2).reshape((3, 2)),
        ...                   y=ivy.arange(3 * 2).reshape((3, 2)))
        >>> c = ivy.Container(x=ivy.arange(2 * 2).reshape((2, 2)),
        ...                   y=ivy.arange(2 * 2).reshape((2, 2)))
        >>> a.multi_dot((b, c))
        {
            x: ivy.array([[26, 49],
                          [80, 148]]),
            y: ivy.array([[26, 49],
                          [80, 148]])
        }
        """
        return self.static_multi_dot(
            (self, *arrays),
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
