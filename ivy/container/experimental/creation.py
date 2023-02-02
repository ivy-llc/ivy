# global
from typing import Optional, Tuple, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithCreationExperimental(ContainerBase):
    @staticmethod
    def static_triu_indices(
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Tuple[ivy.Array]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "triu_indices",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )

    def triu_indices(
        self: ivy.Container,
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Tuple[ivy.Array]] = None,
    ) -> ivy.Container:
        return self.static_triu_indices(
            self,
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )

    @staticmethod
    def static_hann_window(
        window_length: Union[int, ivy.Container],
        periodic: Optional[bool] = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hann_window. This method simply wraps
        the function, and so the docstring for ivy.hann_window also applies to this
        method with minimal changes.

        Parameters
        ----------
        window_length
            container including multiple window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that contains the Hann windows.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_hann(x)
        {
            a: ivy.array([0.0000, 0.7500, 0.7500])
            b: ivy.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hann_window",
            window_length,
            periodic,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hann_window(
        self: ivy.Container,
        periodic: Optional[bool] = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.hann_window. This method simply
        wraps the function, and so the docstring for ivy.hann_window also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container containing the Hann windows.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.hann_window(x)
        {
            a: ivy.array([0.0000, 0.7500, 0.7500])
            b: ivy.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455])
        }
        """
        return self.static_hann_window(self, periodic, dtype, out=out)

    @staticmethod
    def static_kaiser_window(
        window_length: Union[int, ivy.Container],
        periodic: bool = True,
        beta: float = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.kaiser_window. This method
        simply wraps the function, and so the docstring for ivy.kaiser_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        window_length
            input container including window lenghts.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_kaiser_window(x, True, 5)
        {
            a: ivy.array([0.2049, 0.8712, 0.8712]),
            a: ivy.array([0.0367, 0.7753, 0.7753]),
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "kaiser_window",
            window_length,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def kaiser_window(
        self: ivy.Container,
        periodic: bool = True,
        beta: float = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.kaiser_window. This method
        simply wraps the function, and so the docstring for ivy.kaiser_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container including window lenghts.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_kaiser_window(x, True, 5)
        {
            a: ivy.array([0.2049, 0.8712, 0.8712]),
            a: ivy.array([0.0367, 0.7753, 0.7753]),
        }
        """
        return self.static_kaiser_window(
            self,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_kaiser_bessel_derived_window(
        x: Union[int, ivy.Array, ivy.NativeArray, ivy.Container],
        periodic: bool = True,
        beta: float = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.kaiser_bessel_derived_window.
        This method simply wraps the function, and so the docstring for
        ivy.kaiser_bessel_derived_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container including window lenghts.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser Bessel Derived windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_kaiser_bessel_derived_window(x, True, 5)
        {
            a: ivy.array([0.70710677, 0.70710677]),
            b: ivy.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208]),
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "kaiser_bessel_derived_window",
            x,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def kaiser_bessel_derived_window(
        self: ivy.Container,
        periodic: bool = True,
        beta: float = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.kaiser_bessel_derived_window.
        This method simply wraps the function, and so the docstring for
        ivy.kaiser_bessel_derived_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container including window lenghts.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser Bessel Derived windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5))
        >>> x.kaiser_bessel_derived_window(True, 5)
        {
            a: ivy.array([0.70710677, 0.70710677]),
            b: ivy.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208]),
        }
        """
        return self.static_kaiser_bessel_derived_window(
            self,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_hamming_window(
        x: Union[int, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        periodic: Optional[bool] = True,
        alpha: Optional[float] = 0.54,
        beta: Optional[float] = 0.46,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hamming_window.
        This method simply wraps the function, and so the docstring for
        ivy.hamming_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container including window lenghts.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        alpha
            The coefficient alpha in the hamming window equation
        beta
            The coefficient beta in the hamming window equation
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Hamming windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_hamming_window(x, periodic=True, alpha=0.2, beta=2)
        {
            a: ivy.array([-1.8000,  1.2000,  1.2000]),
            b: ivy.array([-1.8000, -0.4180,  1.8180,  1.8180, -0.4180])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hamming_window",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            periodic=periodic,
            alpha=alpha,
            beta=beta,
            dtype=dtype,
            out=out,
        )

    def hamming_window(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        periodic: Optional[bool] = True,
        alpha: Optional[float] = 0.54,
        beta: Optional[float] = 0.46,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.hamming_window.
        This method simply wraps the function, and so the docstring for
        ivy.hamming_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container including window lenghts.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        alpha
            The coefficient alpha in the hamming window equation
        beta
            The coefficient beta in the hamming window equation
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Hamming windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5))
        >>> x.hamming_window(periodic=True, alpha=0.2, beta=2)
        {
            a: ivy.array([-1.8000,  1.2000,  1.2000]),
            b: ivy.array([-1.8000, -0.4180,  1.8180,  1.8180, -0.4180])
        }
        """
        return self.static_hamming_window(
            self, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, out=out
        )

    @staticmethod
    def static_vorbis_window(
        x: Union[int, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.vorbis_window.
        This method simply wraps the function, and so the docstring for
        ivy.vorbis_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container including window lenghts.

        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the vorbis windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_vorbis_window(x)
        {
            a: ivy.array([0., 0.38268343, 0.92387953, 1., 0.92387953,
                          0.38268343]),
            b: ivy.array([0., 0.14943586, 0.51644717, 0.85631905, 0.98877142,
                          1., 0.98877142, 0.85631905, 0.51644717, 0.14943586])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "vorbis_window",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def vorbis_window(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.vorbis_window.
        This method simply wraps the function, and so the docstring for
        ivy.vorbis_window also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container including window lenghts.
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the vorbis windows.

        Examples
        --------
        >>> x = ivy.Container(a=3, b=5))
        >>> x.vorbis_window()
        {
            a: ivy.array([0., 0.38268343, 0.92387953, 1., 0.92387953,
                          0.38268343]),
            b: ivy.array([0., 0.14943586, 0.51644717, 0.85631905, 0.98877142,
                          1., 0.98877142, 0.85631905, 0.51644717, 0.14943586])
        }
        """
        return self.static_vorbis_window(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_tril_indices(
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "tril_indices",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
        )

    def tril_indices(
        self: ivy.Container,
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return self.static_tril_indices(
            self,
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
        )

    @staticmethod
    def static_eye_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.eye_like. This method simply
        wraps the function, and so the docstring for ivy.eye_like also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container from which to derive the output container shape.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``x`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),
                              b=ivy.array([4.5, -5.3, -0, -2.3]))
        >>> y = ivy.Container.static_eye_like(x)
        >>> print(y)
        {
            a: ivy.array([[1.]]),
            b: ivy.array([[1.]])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "eye_like",
            x,
            k=k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    def eye_like(
        self: ivy.Container,
        /,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.eye_like. This method simply
        wraps the function, and so the docstring for ivy.eye_like also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container from which to derive the output container shape.
        k
            index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
            Default: ``0``.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``x`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([3., 8.]), b=ivy.array([2., 2.]))
        >>> y = x.eye_like()
        >>> print(y)
        {
            a: ivy.array([[1.]]),
            b: ivy.array([[1.]])
        }

        """
        return self.static_eye_like(
            self,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )
