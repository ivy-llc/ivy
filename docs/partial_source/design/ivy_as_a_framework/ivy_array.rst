Ivy Array
=========

Here, we explain the :code:`ivy.Array` class, which is the class used to represent all arrays in Ivy. Every Ivy method
returns :code:`ivy.Array` instances for all returned arrays.

The Array Class
---------------

Let’s dive straight in and check out what the :code:`ivy.Array` constructor looks like.

.. code-block:: python

    # ivy/array/__init__.py
    class Array(ivy.ArrayWithDevice, ivy.ArrayWithGeneral,
                ivy.ArrayWithGradients, ivy.ArrayWithImage,
                ivy.ArrayWithLinalg, ivy.ArrayWithLogic,
                ivy.ArrayWithMath, ivy.ArrayWithMeta,
                ivy.ArrayWithRandom, ivy.ArrayWithReductions):

        def __init__(self, data):
            assert ivy.is_native_array(data)
            self._data = data
            self._shape = data.shape
            self._size = functools.reduce(mul,self._data.shape) if len(self._data.shape) >0 else 0
            self._dtype = ivy.dtype(self._data)
            self._device = ivy.dev(data)
            self._dev_str = ivy.dev_to_str(self._device)
            self._pre_repr = 'ivy.'
            if 'gpu' in self._dev_str:
                self._post_repr = ', dev={})'.format(self._dev_str)
            else:
                self._post_repr = ')'

        # Properties #
        # -----------#

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(tuple(self._shape))

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def T(self):
        assert len(self._data.shape) == 2
        return ivy.matrix_transpose(self._data)

    @property
    def mT(self):
        assert len(self._data.shape) >= 2
        return ivy.matrix_transpose(self._data)

We can see that the :code:`ivy.Array` class is a simple wrapper around the :code:`ivy.NativeArray` class, stored in the :code:`self._data` attribute.

This all makes sense, but the first question you might ask is, why do we need a dedicated :code:`ivy.Array` class at all?

Can't we just operate with the native arrays directly such as  :code:`np.ndarray`, :code:`torch.Tensor` etc. when calling ivy methods?

This is a great question, and has many answers with varying importance.
Perhaps the most important motivation for having a dedicated :code:`ivy.Array` class is the unification of array operators,
which we discuss next.

Unifying Operators
------------------

Consider the code below:

.. code-block:: python

    ivy.set_framework(...)
    x = ivy.array([1, 2, 3])
    x[0] = 0
    print(x)

Let's first assume we use numpy in the backend by calling :code:`ivy.set_framework('numpy')` in the first line.

The code will execute without error, printing :code:`array([0, 2, 3])` to the console.

Now consider we use JAX in the backend by calling :code:`ivy.set_framework('jax')` in the first line.

The code will now throw the error :code:`` when we try to set index 0 to the value 0.

The reason for this is that JAX does not support inplace updates for arrays.

This is a problem. The code written above is **pure Ivy code**, which behaves **differently** with different backends.
Therefore, we cannot claim that Ivy code is truly framework-agnostic.

We can re-write the above code as follows:

.. code-block:: python

    ivy.set_framework(...)
    x = ivy.array([1, 2, 3])
    x.__setitem__(0, 0)
    print(x)

If :code:`x` is an :code:`ivy.NativeArray` instance, such as :code:`torch.Tensor` or :code:`np.ndarray`,
then the :code:`__setitem__` method is defined in the native array class, which is completely outside of our control.

However, if :code:`x` is an :code:`ivy.Array` instance then the :code:`__setitem__` method is defined in the :code:`ivy.Array` class, which we have control over.

Let's take a look at how that method is implemented:

.. code-block:: python

    @_native_wrapper
    def __setitem__(self, query, val):
        try:
            self._data.__setitem__(query, val)
        except (AttributeError, TypeError):
            self._data = ivy.scatter_nd(query, val, tensor=self._data, reduction='replace')

We can implement inplace updates in the :code:`ivy.Array` class without requiring inplace updates in the backend array classes.
If the backend does not support inplace updates, then we can use the :code:`ivy.scatter_nd` method to return a new array and store this in the :code:`self._data` attribute.

Now, with :code:`ivy.Array` instances, our code will run without error, regardless of which backend is selected. We can genuinely say our code is fully framework-ag

The same logic applies to all operators. For example, if :code:`x` and :code:`y` are both :code:`ivy.NativeArray` instances then the following code **might** execute identically for all backend frameworks:

.. code-block:: python

    x = ivy.some_method(...)
    y = ivy.some_method(...)
    z = ((x + y) * 3) ** 0.5
    print(z)

Similarly, this code can be rewritten as:

.. code-block:: python

    x = ivy.some_method(...)
    y = ivy.some_method(...)
    z = x.__add__(y).__mul__(3).__pow__(0.5)
    print(z)

Even if this works fine for all backen frameworks now, what if Ivy is updated to support new backends in future, and one of them behaves a little bit differently?
For example, maybe one framework makes the strange decision to return rounded integer data types when integer arrays are raised to floating point powers.

Without enforcing the use of the :code:`ivy.Array` class for arrays returned from Ivy methods, we would have no way to control this behaviour and unify the output :code:`z` for all backends.

Therefore, we have made the decision to require all arrays returned from Ivy methods to be instances of the :code:`ivy.Array` class.


API Monkey Patching
-------------------

If you've looked through the code at all, you may start to notice that the Ivy methods do not seem to return :code:`ivy.Array` instances at all.

This is true, as far as the code appears, the methods actually return :code:`ivy.NativeArray` instances.

For example, let's take a look at the :code:`ivy.inv` method with a PyTorch backend.

.. code-block:: python

    # ivy/functional/backends/torch/linear_algebra.py
    def inv(x: torch.Tensor) -> torch.Tensor:
        return torch.inverse(x)

Passing in an :code:`ivy.Array` instance for :code:`x` will throw an error. The method will also always return a :code:`torch.Tensor`, not an :code:`ivy.Array`,
but we just explained that every Ivy method accepts are returns :code:`ivy.Array` instances, what gives?

In order to avoid excessive code duplication, we have opted to implement this behaviour via monkey-patching, using :code:`_wrap_method`:

.. code-block:: python

    # ivy/func_wrapper.py
    def _wrap_method(fn):

        if hasattr(fn, '__name__') and \
                (fn.__name__[0] == '_' or
                 fn.__name__ in NON_WRAPPED_METHODS):
            return fn

        if hasattr(fn, 'wrapped') and fn.wrapped:
            return fn

        def _method_wrapped(*args, **kwargs):
            native_args, native_kwargs = \
                ivy.args_to_native(*args, **kwargs)
            return ivy.to_ivy(
                        fn(*native_args, **native_kwargs),
                        nested=True)

        if hasattr(fn, '__name__'):
            _method_wrapped.__name__ = fn.__name__
        _method_wrapped.wrapped = True
        _method_wrapped.inner_fn = fn
        return _method_wrapped


Breaking the method down a bit, first we verify the method should be wrapped, otherwise we return the method without wrapping.
Next, we check if the method is already wrapped, and if so we just return this already wrapped method.
Then we define the new wrapped method :code:`_method_wrapped`.
Finally, we copy the method name over to :code:`_method_wrapped`, and flag the wrapped attribute,
store the unwrapped inner function as an attribute, and return the wrapped method.

The unwrap method is much simpler, implemented as follows:

.. code-block:: python

    # ivy/func_wrapper.py
    def _unwrap_method(method_wrapped):

        if not hasattr(method_wrapped, 'wrapped') or \
                not method_wrapped.wrapped:
            return method_wrapped
        return method_wrapped.inner_fn

When setting any backend framework, the entire :code:`ivy.__dict__` is traversed and every method is wrapped using the :code:`_wrap_method` outlined above.
Therefore, all Ivy methods accept and return :code:`ivy.Array` instances without issue, whist still making use of the wrapped backend methods which only operate with :code:`ivy.NativeArray` instances, such as:code:`torch.Tensor` etc.

Importantly, :code:`ivy.args_to_native` does not fail in the inputs are already native arrays.
This means that every Ivy method is able to accept either :code:`ivy.Array` or :code:`ivy.NativeArray`instances for each array argument,
but the returned arrays are always enforced to be :code:`ivy.Array` instances.

This just simplifies the transition from native code to Ivy code in any given project, preventing code such as the following from needlessly throwing an error:

.. code-block:: python

    x = torch.randn(10, 10)
    # lots of torch code
    y = ivy.inv(x)
    # lots of ivy code

Instance Methods
----------------

Taking a look at the class definition, you may wonder why there are so many parent classes!
The only reason the Array class derives from so many different Array classes is so we can compartmentalize the different array functions into separate classes for better code readability.

All methods in the Ivy functional API are implemented as public methods in the :code:`ivy.Array` class via inheritance. For example, a few functions in :code:`ivy.ArrayWithGeneral` are shown below.

.. code-block:: python

    # ivy/array/general.py
    class ArrayWithGeneral(abc.ABC):

        def reshape(self, newshape):
            return ivy.reshape(self, new_shape)

        def transpose(self, axes=None):
            return ivy.transpose(self, axes)

        def flip(self, axis=None, batch_shape=None):
            return ivy.flip(self, axis, batch_shape)

Effectively, the :code:`ivy.Array` class wraps the backend array object, storing it in :code:`self._data`, and this allows all Ivy methods to be called as instance methods of the array. One benefit of this is that it can help to tidy up code. For example:

.. code-block:: python

    x = ivy.ones((1, 2, 3, 4, 5))

    # without ivy.Array
    y = ivy.reshape(ivy.flip(ivy.transpose(
                ivy.reshape(x, (6, 20)), (1, 0)), 0), (2, 10, 6))

    # with ivy.Array
    y = x.reshape((6, 20)).transpose((1, 0)).flip(0).reshape((2, 10, 6))

In the example above, not only is the :code:`ivy.Array` approach shorter to write, but more importantly there is much better alignment between each function and the function arguments. It’s hard to work out which shape parameters align with which method in the first case, but in the second case this is crystal clear.

In addition to the functions in the topic-specific parent classes, there are 41 builtin methods implemented directly in the :code:`ivy.Array` class, each of which directly wrap a method in Ivy's functional API. some examples are given below.

.. code-block:: python

    # ivy/array/__init__.py
    def __add__(self, other):
        return ivy.add(self._data, other)

    def __radd__(self, other):
        return ivy.add(self._data, other)

    def __sub__(self, other):
        return ivy.sub(self._data, other)

    def __rsub__(self, other):
        return ivy.sub(self._data, other)

These enable builtin operations to be performed on the :code:`ivy.Array` instances, and also combinations of :code:`ivy.Array` and :code:`ivy.NativeArray` instances.

.. code-block:: python

    x = ivy.array([0., 1., 2.])
    y = torch.tensor([0., 1., 2.]).cuda()

    assert isinstance(x + y, ivy.Array)
    assert isinstance(y + x, ivy.Array)
    assert isinstance(x - y, ivy.Array)
    assert isinstance(y - x, ivy.Array)



**Round Up**

That should hopefully be enough to get you started with the Ivy Array 😊

Please check out the discussions on the `repo <https://github.com/unifyai/ivy>`_ for FAQs, and reach out on `discord <https://discord.gg/ZVQdvbzNQJ>`_ if you have any questions!
