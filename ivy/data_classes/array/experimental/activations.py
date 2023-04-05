# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithActivationsExperimental(abc.ABC):
    def logit(
        self, /, *, eps: Optional[float] = None, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logit. This method
        simply wraps the function, and so the docstring for ivy.logit
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        eps
            When eps is None the function outpus NaN where x < 0 or x > 1.
            and inf or -inf where x = 1 or x = 0, respectively.
            Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
        out
            Optional output array.

        Returns
        -------
        ret
            Array containing elementwise logits of x.

        Examples
        --------
        >>> x = ivy.array([1, 0, 0.9])
        >>> z = x.logit()
        >>> print(z)
        ivy.array([       inf,       -inf, 2.19722438])

        >>> x = ivy.array([1, 2, -0.9])
        >>> z = x.logit(eps=0.2)
        >>> print(z)
        ivy.array([ 1.38629448,  1.38629448, -1.38629436])

        """
        return ivy.logit(self, eps=eps, out=out)

    def thresholded_relu(
        self: ivy.Array,
        /,
        *,
        threshold: Union[int, float] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.thresholded_relu.
        This method simply wraps the function, and so the docstring
        for ivy.thresholded_relu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        threshold
            threshold value above which the activation is linear. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the relu activation function applied element-wise
            with custom threshold.

        Examples
        --------
        >>> x = ivy.array([-1., .2, 1.])
        >>> y = x.thresholded_relu(threshold=0.5)
        >>> print(y)
        ivy.array([0., 0., 1.])
        """
        return ivy.thresholded_relu(self._data, threshold=threshold, out=out)

    def threshold(
        self: ivy.Array,
        threshold: Union[int, float],
        value: Union[int, float],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.threshold.
        This method simply wraps the function, and so the docstring
        for ivy.threshold also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        threshold
            Threshold value above which the activation is linear.
        value
            The value to replace with
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the threshold applied element-wise
            with custom threshold.

        Examples
        --------
        >>> x = ivy.array([-1., .2, 1.])
        >>> y = x.threshold(0.5, 2.)
        >>> print(y)
        ivy.array([2., 2., 1.])
        """
        return ivy.threshold(self._data, threshold, value, out=out)

    def relu6(self, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """Applies the rectified linear unit 6 function element-wise.

        Parameters
        ----------
        self
            Input array
        out
            Optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array containing the rectified linear unit 6 activation
            of each element in input.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> y = ivy.relu6(x)
        >>> print(y)
        ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])

        >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> y = ivy.zeros(9)
        >>> ivy.relu6(x, out = y)
        >>> print(y)
        ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])

        With :class:`ivy.Container` input:

        >>> x = {
                    a: ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
                    b: ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
                }
        >>> x = ivy.relu6(x, out=x)
        >>> print(x)
        {
        a: ivy.array([0., 0., 0., 0., 1., 2., 3., 4., 5.]),
        b: ivy.array([1., 2., 3., 4., 5., 6., 6., 6., 6.])
        }
        """
        return ivy.relu6(self._data, out=out)

    def batch_norm(
        self,
        mean: Union[ivy.NativeArray, ivy.Array],
        variance: Union[ivy.NativeArray, ivy.Array],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        training: bool = False,
        eps: float = 1e-5,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.batch_norm. This method
        simply wraps the function, and so the docstring for ivy.batch_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of shape (N,C,S), where N is the batch dimension, C is the
            feature dimension and S corresponds to the following spatial dimensions.
        mean
            A mean array for the input's normalization.
        variance
            A variance array for the input's normalization.
        offset
            An offset array. If present, will be added to the normalized input.
        scale
            A scale array. If present, the scale is applied to the normalized input.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.

        Returns
        -------
        ret
             Array containing the normalized, scaled, offset values.
        """
        return ivy.batch_norm(
            self._data,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
        )

    def group_norm(
        self,
        num_groups: int,
        /,
        *,
        weight: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        bias: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        eps: float = 1e-5,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.group_norm. This method
        simply wraps the function, and so the docstring for ivy.group_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of shape (N,C,∗), where N is the batch dimension, C is the
            feature dimension.
        num_groups
            Number of groups to separate the channels into
        weight
            A scale array. If present, the scale is applied to the normalized input.
        bias
            An offset array. If present, will be added to the normalized input.
        eps
            A small float number to avoid dividing by 0.

        Returns
        -------
        ret
             Array containing the normalized, scaled, offset values.
        """
        return ivy.group_norm(
            self._data,
            num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    def logsigmoid(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.logsigmoid. This method
        simply wraps the function, and so the docstring for ivy.logsigmoid
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.

        Returns
        -------
            Array with same shape as input with Log-sigmoid applied to every element.

        Examples
        --------
        >>> x = ivy.array([-1., 2., 4., -10.])
        >>> z = x.logsigmoid()
        >>> print(z)
        ivy.array([ -1.31326175,  -0.126928  ,  -0.01814993, -10.00004578])

        >>> x = ivy.array([-2.5, 1., 0, 4.5])
        >>> z = x.logsigmoid())
        >>> print(z)
        ivy.array([-2.57888985, -0.31326169, -0.69314718, -0.01104775])

        """
        return ivy.logsigmoid(self._data, out=out)

    def hard_tanh(
        self: ivy.Array,
        /,
        *,
        min_value: float = -1.0,
        max_value: float = 1.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hard_tanh.
        This method simply wraps the function, and so the docstring for ivy.hard_tanh
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array
        out
            Optional output array for writing the result to. It must have the same shape
            the input broadcast to default: None

        Returns
        -------
        ret
            An array with the hard tanh activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-1. ,  1. ,  0.1])
        >>> y = x.hard_tanh()
        >>> print(y)
        ivy.array([-1. ,  1. ,  0.1])
        """
        return ivy.hard_tanh(
            self._data, min_value=min_value, max_value=max_value, out=out
        )

    def softsign(self, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softsign. This method simply wraps the
        function, and so the docstring for ivy.softsign also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Optional output array, for writing the result to. It must have a shape

        Returns
        -------
        ret
            An array with the softsign activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softsign()
        >>> print(y)
        ivy.array([-0.25711316, -0.39360863])
        """
        return ivy.softsign(self._data, out=out)

    def silu(self, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.silu. This method simply wraps the
        function, and so the docstring for ivy.silu also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Optional output array, for writing the result to. It must have a shape

        Returns
        -------
        ret
            An array with the silu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.silu()
        >>> print(y)
        ivy.array([-0.14339909, -0.22276618])
        """
        return ivy.silu(self._data, out=out)

    def hard_silu(self, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hard_silu. This method simply wraps the
        function, and so the docstring for ivy.hard_silu also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Optional output array, for writing the result to. It must have a shape

        Returns
        -------
        ret
            An array with the hard silu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.hard_silu()
        >>> print(y)
        ivy.array([-0.1530858 , -0.25432822])
        """
        return ivy.hard_silu(self._data, out=out)

    def elu(
        self: ivy.Array,
        /,
        *,
        alpha: float = 1.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.elu. This method simply wraps
        the function, and so the docstring for ivy.elu also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        alpha
            Scalar or array of alpha values
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.elu()
        >>> print(y)
        ivy.array([ 0.39     , -0.5725851])

        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.elu(alpha=2.)
        >>> print(y)
        ivy.array([ 0.39     , -1.1451702])

        """
        return ivy.elu(self._data, alpha=alpha, out=out)

    def celu(
        self: ivy.Array,
        /,
        *,
        alpha: float = 1.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.celu. This method simply wraps
        the function, and so the docstring for ivy.celu also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        alpha
            The slope of the negative section.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the continuously-differentiable exponential linear
            unit activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.celu()
        >>> print(y)
        ivy.array([ 0.39, -0.5725851])

        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.celu(alpha=2.0)
        >>> print(y)
        ivy.array([ 0.39, -0.6924603])

        """
        return ivy.celu(self._data, alpha=alpha, out=out)

    def parametric_relu(
        self: ivy.Array,
        weight: Union[float, ivy.Array],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        It takes input data (Array) and slope array as input,
        and produces one output data (array) where the function
        f(x) = slope * x for x < 0, f(x) = x for x >= 0., is applied
        to the data array elementwise. This operator supports unidirectional
        broadcasting (array slope should be unidirectional broadcastable to
        input tensor X);

        Parameters
        ----------
        self
            Input array.
        slope
            Slope Array. The shape of slope can be smaller than first input X;
            if so, its shape must be unidirectional broadcastable to X.
        out
            Optional output array.

        Returns
        -------
        ret
            Input array with parametric relu applied elementwise.
        """

        return ivy.parametric_relu(self._data, weight, out=out)

    def hard_sigmoid(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hard_sigmoid. This method simply wraps
        the function, and so the docstring for ivy.hard_sigmoid also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        alpha
            The slope of the negative section.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the hard_sigmoid activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.hard_sigmoid()
        >>> print(y)
        ivy.array([0.565     , 0.35833335])
        """
        return ivy.hard_sigmoid(self._data, out=out)

    def selu(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.selu. This method simply wraps
        the function, and so the docstring for ivy.selu also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the selu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.selu()
        >>> print(y)
        ivy.array([ 0.41, -1.0])
        """
        return ivy.selu(self._data, out=out)

    def hardshrink(
        self: ivy.Array,
        /,
        *,
        lambd: Optional[Union[int, float]] = 0.5,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hardshrink. This method simply wraps
        the function, and so the docstring for ivy.hardshrink also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        lambd
            The value where the function is zero for inputs that are absolute value
            less than it.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the selu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
        >>> y = x.hardshrink()
        >>> print(y)
        ivy.array([ 0.69, -0.85,  0.,  0.])

        >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
        >>> y = x.hardshrink(lambd=0.2)
        >>> print(y)
        ivy.array([ 0.69, -0.85,  0.4,  0.])

        """
        return ivy.hardshrink(self._data, lambd=lambd, out=out)

    def softshrink(
        self: ivy.Array,
        /,
        *,
        lambd: Optional[Union[int, float]] = 0.5,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softshrink. This method simply wraps
        the function, and so the docstring for ivy.softshrink also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        lambd
            The value where the function is zero for inputs that are absolute value
            less than it. It must be no less than zero.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the selu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
        >>> y = x.softshrink()
        >>> print(y)
        ivy.array([ 0.19, -0.35,  0.,  0.])

        >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
        >>> y = x.softshrink(lambd=0.2)
        >>> print(y)
        ivy.array([ 0.49, -0.65,  0.2,  0.])

        """
        return ivy.softshrink(self._data, lambd=lambd, out=out)

    def glu(self, /, *, axis: int = -1, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.glu. This method simply wraps the
        function, and so the docstring for ivy.glu also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            the axis along which the split should be computed
        out
            optional output array, for writing the result to. It must have a shape

        Returns
        -------
        ret
            an array with the glu activation function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.glu()
        >>> print(y)
        ivy.array([-0.11877888])
        """
        return ivy.glu(self._data, axis=axis, out=out)
