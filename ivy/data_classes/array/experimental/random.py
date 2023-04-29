# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithRandomExperimental(abc.ABC):
    def dirichlet(
        self: ivy.Array,
        /,
        *,
        size: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.dirichlet. This method simply wraps the
        function, and so the docstring for ivy.shuffle also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Sequence of floats of length k
        size
            optional int or tuple of ints, Output shape. If the given shape is,
            e.g., (m, n), then m * n * k samples are drawn. Default is None,
            in which case a vector of length k is returned.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The drawn samples, of shape (size, k).

        Examples
        --------
        >>> alpha = ivy.array([1.0, 2.0, 3.0])
        >>> alpha.dirichlet()
        ivy.array([0.10598304, 0.21537054, 0.67864642])

        >>> alpha = ivy.array([1.0, 2.0, 3.0])
        >>> alpha.dirichlet(size = (2,3))
        ivy.array([[[0.48006698, 0.07472073, 0.44521229],
            [0.55479872, 0.05426367, 0.39093761],
            [0.19531053, 0.51675832, 0.28793114]],

        [[0.12315625, 0.29823365, 0.5786101 ],
            [0.15564976, 0.50542368, 0.33892656],
            [0.1325352 , 0.44439589, 0.42306891]]])
        """
        return ivy.dirichlet(self, size=size, dtype=dtype, seed=seed, out=out)

    def beta(
        self: ivy.Array,
        alpha: Union[int, ivy.Array, ivy.NativeArray],
        beta: Union[int, ivy.Array, ivy.NativeArray],
        /,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.beta. This method simply wraps the
        function, and so the docstring for ivy.beta also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input Array.
        alpha
            The first parameter of the beta distribution.
        beta
            The second parameter of the beta distribution.
        device
            device on which to create the array.
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized beta distribution with the shape of
            the array.
        """
        return ivy.beta(
            alpha,
            beta,
            shape=self.shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def gamma(
        self: ivy.Array,
        alpha: Union[int, ivy.Array, ivy.NativeArray],
        beta: Union[int, ivy.Array, ivy.NativeArray],
        /,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gamma. This method simply wraps the
        function, and so the docstring for ivy.gamma also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input Array.
        alpha
            The first parameter of the gamma distribution.
        beta
            The second parameter of the gamma distribution.
        device
            device on which to create the array.
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized gamma distribution with the shape of
            the input array.
        """
        return ivy.gamma(
            alpha,
            beta,
            shape=self.shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def poisson(
        self: ivy.Array,
        *,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ):
        """
        Parameters
        ----------
        self
            Input Array of rate paramter(s). It must have a shape that is broadcastable
            to the requested shape
        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'ivy.shape(lam)' samples are drawn)
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized poisson distribution.

        Examples
        --------
        >>> lam = ivy.array([1.0, 2.0, 3.0])
        >>> lam.poisson()
        ivy.array([1., 4., 4.])

        >>> lam = ivy.array([1.0, 2.0, 3.0])
        >>> lam.poisson(shape=(2,3))
        ivy.array([[0., 2., 2.],
                   [1., 2., 3.]])
        """
        return ivy.poisson(
            self,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def bernoulli(
        self: ivy.Array,
        *,
        logits: Optional[Union[float, ivy.Array, ivy.NativeArray]] = None,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ):
        """

        Parameters
        ----------
        self
             An N-D Array representing the probability of a 1 event.
             Each entry in the Array parameterizes an independent Bernoulli
             distribution. Only one of logits or probs should be passed in
        logits
            An N-D Array representing the log-odds of a 1 event.
            Each entry in the Array parameterizes an independent Bernoulli
            distribution where the probability of an event is sigmoid
            (logits). Only one of logits or probs should be passed in.

        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'ivy.shape(logits)' samples are drawn)
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).

        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the Bernoulli distribution
        """
        return ivy.bernoulli(
            self,
            logits=logits,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )
