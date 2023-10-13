# global
import abc
from typing import Optional, Union, Literal

# local
import ivy


# ToDo: implement all methods here as public instance methods


class _ArrayWithActivations(abc.ABC):
    def relu(
        self: ivy.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.relu. This method simply
        wraps the function, and so the docstring for ivy.relu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the relu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-1., 0., 1.])
        >>> y = x.relu()
        >>> print(y)
        ivy.array([0., 0., 1.])
        """
        return ivy.relu(self._data, complex_mode=complex_mode, out=out)

    def leaky_relu(
        self: ivy.Array,
        /,
        *,
        alpha: float = 0.2,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.leaky_relu. This method
        simply wraps the function, and so the docstring for ivy.leaky_relu also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        alpha
            the slope of the negative section.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the leaky relu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.leaky_relu()
        >>> print(y)
        ivy.array([ 0.39, -0.17])
        """
        return ivy.leaky_relu(
            self._data, alpha=alpha, complex_mode=complex_mode, out=out
        )

    def gelu(
        self: ivy.Array,
        /,
        *,
        approximate: bool = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.gelu. This method simply
        wraps the function, and so the docstring for ivy.gelu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        approximate
            whether to use the approximate version of the gelu function.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the gelu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-1.2, -0.6, 1.5])
        >>> y = x.gelu()
        >>> print(y)
        ivy.array([-0.138, -0.165, 1.4])
        """
        return ivy.gelu(
            self._data, approximate=approximate, complex_mode=complex_mode, out=out
        )

    def sigmoid(
        self: ivy.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.sigmoid.

        This method simply wraps the function, and so the docstring for ivy.sigmoid also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array for writing the result to. It must have the same shape
            the input broadcast to default: None

        Returns
        -------
        ret
            an array with the sigmoid activation function applied element-wise.


        Examples
        --------
        >>> x = ivy.array([-1., 1., 2.])
        >>> y = x.sigmoid()
        >>> print(y)
        ivy.array([0.269, 0.731, 0.881])
        """
        return ivy.sigmoid(self._data, complex_mode=complex_mode, out=out)

    def softmax(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.softmax. This method simply
        wraps the function, and so the docstring for ivy.softmax also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            the axis or axes along which the softmax should be computed
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the softmax activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([1.0, 0, 1.0])
        >>> y = x.softmax()
        >>> print(y)
        ivy.array([0.422, 0.155, 0.422])
        """
        return ivy.softmax(self._data, axis=axis, complex_mode=complex_mode, out=out)

    def softplus(
        self: ivy.Array,
        /,
        *,
        beta: Optional[Union[int, float]] = None,
        threshold: Optional[Union[int, float]] = None,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.softplus. This method
        simply wraps the function, and so the docstring for ivy.softplus also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        beta
            the beta parameter of the softplus function.
        threshold
            the threshold parameter of the softplus function.
        complex_mode
           optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape

        Returns
        -------
        ret
            an array with the softplus activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softplus()
        >>> print(y)
        ivy.array([0.535,0.42])

        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softplus(beta=0.5)
        >>> print(y)
        ivy.array([1.22, 1.09])

        >>> x = ivy.array([1.31, 2., 2.])
        >>> y = x.softplus(threshold=2, out=x)
        >>> print(x)
        ivy.array([1.55, 2.13, 2.13])
        """
        return ivy.softplus(
            self._data,
            beta=beta,
            threshold=threshold,
            complex_mode=complex_mode,
            out=out,
        )

    def log_softmax(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = -1,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.log_softmax. This method
        simply wraps the function, and so the docstring for ivy.log_softmax
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            the axis or axes along which the log_softmax should be computed
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the log_softmax activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-1.0, -0.98, 2.3])
        >>> y = x.log_softmax()
        >>> print(y)
        ivy.array([-3.37, -3.35, -0.0719])

        >>> x = ivy.array([2.0, 3.4, -4.2])
        >>> y = x.log_softmax(x)
        ivy.array([-1.62, -0.221, -7.82 ])
        """
        return ivy.log_softmax(
            self._data,
            axis=axis,
            complex_mode=complex_mode,
            out=out,
        )

    def mish(
        self: ivy.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.mish. This method simply
        wraps the function, and so the docstring for ivy.mish also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Examples
        --------
        >>> x = ivy.array([-1., 0., 1.])
        >>> y = x.mish()
        >>> print(y)
        ivy.array([-0.30340147,  0.        ,  0.86509842])
        """
        return ivy.mish(self._data, complex_mode=complex_mode, out=out)

    def hardswish(
        self: ivy.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Apply the hardswish activation function element-wise.

        Parameters
        ----------
        x
            input array
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hardswish activation of each element in ``x``.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([0., 0., 4.])
        >>> y = ivy.hardswish(x)
        >>> y
        ivy.array([0., 0., 4.])

        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-3., 4., 5.]), b=ivy.array([0., 5.]))
        >>> x = ivy.hardswish(x, out=x)
        >>> x
        {
            a: ivy.array([-0.,  4.,  5.]),
            b: ivy.array([0., 5.])
        }
        """
        return ivy.hardswish(self._data, complex_mode=complex_mode, out=out)
