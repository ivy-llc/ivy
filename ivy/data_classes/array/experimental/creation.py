# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithCreationExperimental(abc.ABC):
    def eye_like(
        self: ivy.Array,
        /,
        *,
        k: int = 0,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.eye_like. This method simply wraps the
        function, and so the docstring for ivy.eye_like also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        k
            index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
            Default: ``0``.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = ivy.array([[2, 3, 8],[1, 2, 1]])
        >>> y = x.eye_like()
        >>> print(y)
        ivy.array([[1., 0., 0.],
                    0., 1., 0.]])
        """
        return ivy.eye_like(self._data, k=k, dtype=dtype, device=device, out=out)

    def unsorted_segment_min(
        self: ivy.Array,
        segment_ids: ivy.Array,
        num_segments: Union[int, ivy.Array],
    ) -> ivy.Array:
        r"""
        ivy.Array instance method variant of ivy.unsorted_segment_min. This method
        simply wraps the function, and so the docstring for ivy.unsorted_segment_min
        also applies to this method with minimal changes.

        Note
        ----
        If the given segment ID `i` is negative, then the corresponding
        value is dropped, and will not be included in the result.

        Parameters
        ----------
        self
            The array from which to gather values.

        segment_ids
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.

        num_segments
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret
            The output array, representing the result of a segmented min operation.
            For each segment, it computes the min value in `self` where `segment_ids`
            equals to segment ID.
        """
        return ivy.unsorted_segment_min(self._data, segment_ids, num_segments)

    def unsorted_segment_sum(
        self: ivy.Array,
        segment_ids: ivy.Array,
        num_segments: Union[int, ivy.Array],
    ) -> ivy.Array:
        r"""
        ivy.Array instance method variant of ivy.unsorted_segment_sum. This method
        simply wraps the function, and so the docstring for ivy.unsorted_segment_sum
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The array from which to gather values.

        segment_ids
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.

        num_segments
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret
            The output array, representing the result of a segmented sum operation.
            For each segment, it computes the sum of values in `self` where
            `segment_ids` equals to segment ID.
        """
        return ivy.unsorted_segment_sum(self._data, segment_ids, num_segments)

      
    def blackman_window(
        self: ivy.Array,
        /,
        *,
        periodic: bool = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.blackman_window. This method simply
        wraps the function, and so the docstring for ivy.blackman_window also applies to
        this method with minimal changes.
        
        Parameters
        ----------
        self
            int.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
            Default: ``True``.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.
        Returns
        -------
        ret
            The array containing the window.
        Examples
        --------
        >>> ivy.blackman_window(4, periodic = True)
        ivy.array([-1.38777878e-17,  3.40000000e-01,  1.00000000e+00,  3.40000000e-01])
        >>> ivy.blackman_window(7, periodic = False)
        ivy.array([-1.38777878e-17,  1.30000000e-01,  6.30000000e-01,  1.00000000e+00,
        6.30000000e-01,  1.30000000e-01, -1.38777878e-17])
        """
        return ivy.blackman_window(
            self._data, periodic=periodic, dtype=dtype, device=device, out=out
        )
      
    def trilu(
        self: ivy.Array,
        /,
        *,
        k: int = 0,
        upper: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.trilu. This method simply wraps the
        function, and so the docstring for ivy.trilu also applies to this method with
        minimal changes.
        
        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.    *,
        k
            diagonal below or above which to zero elements. If k = 0, the diagonal is
            the main diagonal. If k < 0, the diagonal is below the main diagonal. If
            k > 0, the diagonal is above the main diagonal. Default: ``0``.
        upper
            indicates whether upper or lower part of matrix is retained.
            Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the upper triangular part(s). The returned array must
            have the same shape and data type as ``self``. All elements below the
            specified diagonal k must be zeroed. The returned array should be allocated
            on the same device as ``self``.
        """
        return ivy.trilu(self._data, k=k, upper=upper, out=out)

    def mel_weight_matrix(
        self: ivy.Array,
        num_mel_bins: Union[int, ivy.Array],
        dft_length: Union[int, ivy.Array],
        sample_rate: Union[int, ivy.Array],
        lower_edge_hertz: Optional[Union[float, ivy.Array]] = 0.0,
        upper_edge_hertz: Optional[Union[float, ivy.Array]] = 3000.0,
    ):
        """
        Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a
        linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins
        frequency information based on the [lower_edge_hertz, upper_edge_hertz]
        range on the mel scale. This function defines the mel scale in terms of a frequency
        in hertz according to the following formula: mel(f) = 2595 * log10(1 + f/700)
        Parameters
        ----------
        num_mel_bins
            The number of bands in the mel spectrum.
        dft_length
            The size of the original DFT obtained from (n_fft / 2 + 1).
        sample_rate
            Samples per second of the input signal.
        lower_edge_hertz
            Lower bound on the frequencies to be included in the mel spectrum.
        upper_edge_hertz
            The desired top edge of the highest frequency band.
        Returns
        -------
        ret
            MelWeightMatrix of shape:  [frames, num_mel_bins].
        Examples
        --------
        >>> x = ivy.array([[1, 2, 3],
        >>>                [1, 1, 1],
        >>>                [5,6,7  ]])
        >>> x.mel_weight_matrix(3, 3, 8000)
        ivy.array([[0.        ,0.        , 0.],
                  [0.        ,0. , 0.75694758],
                  [0.        ,0. , 0.       ]])
        """
        return ivy.mel_weight_matrix(
            num_mel_bins,
            dft_length,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )
