Ivy Array
=========

Here, we explain the :code:`ivy.Array` class, which is the class used to represent all arrays in Ivy. Every Ivy method
returns :code:`ivy.Array` instances for all returned arrays.

The Array Class
---------------

Let’s dive straight in and check out what the :code:`ivy.Array` constructor looks like.

.. code-block:: python

    # ivy/array/__init__.py
    class Array(
        ArrayWithActivations,
        ArrayWithCreation,
        ArrayWithDataTypes,
        ArrayWithDevice,
        ArrayWithElementwise,
        ArrayWithGeneral,
        ArrayWithGradients,
        ArrayWithImage,
        ArrayWithLayers,
        ArrayWithLinearAlgebra,
        ArrayWithLosses,
        ArrayWithManipulation,
        ArrayWithNorms,
        ArrayWithRandom,
        ArrayWithSearching,
        ArrayWithSet,
        ArrayWithSorting,
        ArrayWithStatistical,
        ArrayWithUtility,
    ):
        def __init__(self, data):
            ArrayWithActivations.__init__(self)
            ArrayWithCreation.__init__(self)
            ArrayWithDataTypes.__init__(self)
            ArrayWithDevice.__init__(self)
            ArrayWithElementwise.__init__(self)
            ArrayWithGeneral.__init__(self)
            ArrayWithGradients.__init__(self)
            ArrayWithImage.__init__(self)
            ArrayWithLayers.__init__(self)
            ArrayWithLinearAlgebra.__init__(self)
            ArrayWithLosses.__init__(self)
            ArrayWithManipulation.__init__(self)
            ArrayWithNorms.__init__(self)
            ArrayWithRandom.__init__(self)
            ArrayWithSearching.__init__(self)
            ArrayWithSet.__init__(self)
            ArrayWithSorting.__init__(self)
            ArrayWithStatistical.__init__(self)
            ArrayWithUtility.__init__(self)
            self._init(data)

        def _init(self, data):
            if ivy.is_ivy_array(data):
                self._data = data.data
            else:
                assert ivy.is_native_array(data)
                self._data = data
            self._shape = self._data.shape
            self._size = (
                functools.reduce(mul, self._data.shape) if len(self._data.shape) > 0 else 0
            )
            self._dtype = ivy.dtype(self._data)
            self._device = ivy.dev(self._data)
            self._dev_str = ivy.as_ivy_dev(self._device)
            self._pre_repr = "ivy."
            if "gpu" in self._dev_str:
                self._post_repr = ", dev={})".format(self._dev_str)
            else:
                self._post_repr = ")"
            self.framework_str = ivy.current_backend_str()
            self._is_variable = ivy.is_variable(self._data)

        # Properties #
        # -----------#

        # noinspection PyPep8Naming
        @property
        def mT(self):
            assert len(self._data.shape) >= 2
            return ivy.matrix_transpose(self._data)

        @property
        def data(self):
            return self._data

        @property
        def shape(self):
            return ivy.Shape(self._shape)

We can see that the :code:`ivy.Array` class is a simple wrapper around an :code:`ivy.NativeArray` class (such as  :code:`np.ndarray`, :code:`torch.Tensor` etc), stored in the :code:`self._data` attribute.

This all makes sense, but the first question you might ask is, why do we need a dedicated :code:`ivy.Array` class at all?

Can't we just operate with the native arrays directly such as  :code:`np.ndarray`, :code:`torch.Tensor` etc. when calling ivy methods?

This is a great question, and has a couple of answers with varying importance.
Perhaps the most important motivation for having a dedicated :code:`ivy.Array` class is the unification of array operators,
which we discuss next!

Unifying Operators
------------------

Let's assume that there is no such thing as the :code:`ivy.Array` class,
and we are just returning native arrays from all Ivy methods.

Consider the code below:

.. code-block:: python

    ivy.set_backend(...)
    x = ivy.array([1, 2, 3])
    x[0] = 0
    print(x)

Let's first assume we use numpy in the backend by calling :code:`ivy.set_backend('numpy')` in the first line.
:code:`x` would then be a :code:`np.ndarray` instance.

In this case, the code will execute without error, printing :code:`array([0, 2, 3])` to the console.

Now consider we use JAX in the backend by calling :code:`ivy.set_backend('jax')` in the first line.
:code:`x` would then be a :code:`jax.numpy.ndarray` instance.

The code will now throw the error :code:`TypeError: '<class 'jaxlib.xla_extension.DeviceArray'>' object does not support item assignment.` :code:`JAX arrays are immutable.` :code:`Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method` when we try to set index 0 to the value 0.

As can be seen from the error message, the reason for this is that JAX does not support inplace updates for arrays.

This is a problem. The code written above is **pure Ivy code** which means it should behave identically irrespective of the backend, but as we've just seen it behaves **differently** with different backends.
Therefore, in this case, we could not claim that the Ivy code was truly framework-agnostic.

For the purposes of explanation, we can re-write the above code as follows:

.. code-block:: python

    ivy.set_backend(...)
    x = ivy.array([1, 2, 3])
    x.__setitem__(0, 0)
    print(x)

If :code:`x` is an :code:`ivy.NativeArray` instance, such as :code:`torch.Tensor` or :code:`np.ndarray`,
then the :code:`__setitem__` method is defined in the native array class, which is completely outside of our control.

However, if :code:`x` is an :code:`ivy.Array` instance then the :code:`__setitem__` method is defined in the :code:`ivy.Array` class, which we do have control over.

Let's take a look at how that method is implemented in the :code:`ivy.Array` class:

.. code-block:: python

    @_native_wrapper
    def __setitem__(self, query, val):
        try:
            self._data.__setitem__(query, val)
        except (AttributeError, TypeError):
            self._data = ivy.scatter_nd(
                query, val, tensor=self._data, reduction="replace"
            )._data
            self._dtype = ivy.dtype(self._data)

We can implement inplace updates in the :code:`ivy.Array` class without requiring inplace updates in the backend array classes.
If the backend does not support inplace updates, then we can use the :code:`ivy.scatter_nd` method to return a new array and store this in the :code:`self._data` attribute.

Now, with :code:`ivy.Array` instances, our code will run without error, regardless of which backend is selected. We can genuinely say our code is fully framework-agnostic.

The same logic applies to all python operators. For example, if :code:`x` and :code:`y` are both :code:`ivy.NativeArray` instances then the following code **might** execute identically for all backend frameworks:

.. code-block:: python

    x = ivy.some_method(...)
    y = ivy.some_method(...)
    z = ((x + y) * 3) ** 0.5
    print(z)

Similarly, for demonstration purposes, this code can be rewritten as:

.. code-block:: python

    x = ivy.some_method(...)
    y = ivy.some_method(...)
    z = x.__add__(y).__mul__(3).__pow__(0.5)
    print(z)

Even if this works fine for all backend frameworks now, what if Ivy is updated to support new backends in future, and one of them behaves a little bit differently?
For example, maybe one framework makes the strange decision to return rounded integer data types when integer arrays are raised to floating point powers.

Without enforcing the use of the :code:`ivy.Array` class for arrays returned from Ivy methods, we would have no way to control this behaviour and unify the output :code:`z` for all backends.

Therefore, with the design of Ivy, we have made the decision to require all arrays returned from Ivy methods to be instances of the :code:`ivy.Array` class.

API Monkey Patching
-------------------

All ivy functions with array inputs/outputs have been wrapped to return :code:`ivy.Array` instances while accepting both :code:`ivy.Array` and :code:`ivy.NativeArray` instances.
This allows for the control required to provide a unified array interface.
For more detail on wrapping, see the the `Function Wrapping <https://lets-unify.ai/ivy/deep_dive/3_function_wrapping.html>`_ page in deep dive.


Instance Methods
----------------

Taking a look at the class definition, you may wonder why there are so many parent classes!
The only reason the Array class derives from so many different Array classes is so we can compartmentalize the different array functions into separate classes for better code readability.

All methods in the Ivy functional API are implemented as public instance methods in the :code:`ivy.Array` class via inheritance. For example, a few functions in :code:`ivy.ArrayWithGeneral` are shown below.

.. code-block:: python

    # ivy/array/general.py
    class ArrayWithGeneral(abc.ABC):

        def reshape(self, newshape):
            return ivy.reshape(self, new_shape)

        def transpose(self, axes=None):
            return ivy.transpose(self, axes)

        def flip(self, axis=None, batch_shape=None):
            return ivy.flip(self, axis, batch_shape)

One benefit of these instance methods is that they can help to tidy up code. For example:

.. code-block:: python

    x = ivy.ones((1, 2, 3, 4, 5))

    # without ivy.Array
    y = ivy.reshape(ivy.flip(ivy.transpose(
                ivy.reshape(x, (6, 20)), (1, 0)), 0), (2, 10, 6))

    # with ivy.Array
    y = x.reshape((6, 20)).transpose((1, 0)).flip(0).reshape((2, 10, 6))

In the example above, not only is the :code:`ivy.Array` approach shorter to write, but more importantly there is much better alignment between each function and the function arguments. It’s hard to work out which shape parameters align with which method in the first case, but in the second case this is crystal clear.

In addition to the functions in the topic-specific parent classes, there are about 50 builtin methods implemented directly in the :code:`ivy.Array` class, most of which directly wrap a method in Ivy's functional API. some examples are given below.

.. code-block:: python

    # ivy/array/__init__.py
    def __add__(self, other):
        return ivy.add(self, other)

    def __sub__(self, other):
        return ivy.sub(self, other)

    def __mul__(self, other):
        return ivy.mul(self, other)


**Round Up**

That should hopefully be enough to get you started with the Ivy Array 😊

Please check out the discussions on the `repo <https://github.com/unifyai/ivy>`_ for FAQs, and reach out on `discord <https://discord.gg/ZVQdvbzNQJ>`_ if you have any questions!
