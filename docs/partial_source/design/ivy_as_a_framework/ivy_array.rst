Ivy Array
=========

Here, we explain the :code:`ivy.Array` class, which is the class used to represent all arrays in Ivy. Every Ivy method
returns :code:`ivy.Array` instances to represent the returned arrays.

Without further ado, let’s walk through what the Ivy Array has to offer!

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
            assert ivy.is_array(data)
            self._data = data
            self._shape = data.shape
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
            return self._shape

        @property
        def dtype(self):
            return self._dtype

        @property
        def device(self):
            return self._device

The only reason the Array class derives from so many different Array classes is so we can compartmentalize the different array functions into separate classes for better code readability.

All methods in the Ivy functional API are implemented as public methods in the :code:`ivy.Array` class via inheritance. For example, a few functions in :code:`ivy.ArrayWithGeneral` are shown below.

.. code-block:: python

    # ivy/array/general.py
    class ArrayWithGeneral(abc.ABC):

        def reshape(self, newshape):
            return ivy.to_ivy(ivy.reshape(
                        self._data, new_shape))

        def transpose(self, axes=None):
            return ivy.to_ivy(ivy.transpose(
                        self._data, axes))

        def flip(self, axis=None, batch_shape=None):
            return ivy.to_ivy(ivy.flip(
                        self._data, axis, batch_shape))

Effectively, the :code:`ivy.Array` class wraps the backend array object, storing it in :code:`self._data`, and this allows all Ivy methods to be called as attributes of the array. The method :code:`ivy.to_ivy` recursively converts all :code:`ivy.NativeArray` instances (i.e. :code:`torch.Tensor`)to :code:`ivy.Array` instances. One benefit of the :code:`ivy.Array` class is that it can help to tidy up code. For example:

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
        other = to_native(other)
        res = ivy.add(self._data, other)
        return to_ivy(res)

    def __radd__(self, other):
        other = to_native(other)
        res = ivy.radd(self._data, other)
        return to_ivy(res)

    def __sub__(self, other):
        other = to_native(other)
        res = ivy.sub(self._data, other)
        return to_ivy(res)

    def __rsub__(self, other):
        other = to_native(other)
        res = ivy.rsub(self._data, other)
        return to_ivy(res)

These enable builtin operations to be performed on the :code:`ivy.Array` instances, and also combinations of :code:`ivy.Array` and :code:`ivy.NativeArray` instances.

.. code-block:: python

    x = ivy.array([0., 1., 2.])
    y = torch.tensor([0., 1., 2.]).cuda()

    assert isinstance(x + y, ivy.Array)
    assert isinstance(y + x, ivy.Array)
    assert isinstance(x - y, ivy.Array)
    assert isinstance(y - x, ivy.Array)

Returning Ivy Arrays
--------------------

“But how do we call the backend framework methods when we're using ivy.Array classes?”, I hear you ask.

“:code:`torch.reshape` only accepts :code:`torch.Tensor` instances as input, we can’t just decide to pass in an :code:`ivy.Array` instead”

This is absolutely correct! The following code throws an error.

.. code-block:: python

    x = ivy.array([0., 1., 2.])
    y = torch.reshape(x, (1, 3, 1))

    ->          y = torch.reshape(x, (1, 3, 1))
    -> TypeError: no implementation found for 'torch.reshape' on
    -> types that implement __torch_function__: [0x7fef01e65ad0]

Furthermore, even if it could accept :code:`ivy.Array` instances as input, :code:`torch.reshape` returns a :code:`torch.Tensor`, and so we lose the :code:`ivy.Array` structure of our array as soon as any backend-specific method is called upon it.

This is indeed the case, and in order to run the above code without throwing an error, we would need to write something like the following:

.. code-block:: python

    x = ivy.array([0., 1., 2.])
    x_native = x.data
    y_native = torch.reshape(x, (1, 3, 1))
    y = ivy.Array(y_native)

In general, if integrating ivy code alongside native code, then conversions such as these will be needed between the adjacent blocks of code.

"What about ivy methods? How do they call the backend methods without error?" I also hear you ask.

This is a great question!

All Ivy methods in the functional API are automatically wrapped such that all inputs are recursively parsed to convert :code:`ivy.Array` instances to :code:`ivy.NativeArray` instances (i.e. :code:`torch.Tensor`), then the backend method is called as usual, and finally the return values are recursively parsed to convert all :code:`ivy.NativeArray` instances into :code:`ivy.Array` instances. The wrapping method is implemented as follows:

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

First, we verify the method should be wrapped, otherwise we return the method without wrapping.
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

When setting any backend framework, the entire :code:`ivy.__dict__` is traversed and all methods are wrapped using the :code:`_wrap_method` outlined above. Therefore, all Ivy methods operate as expected, accepting and returning :code:`ivy.Array` instances without issue, whist still making use of the wrapped backend methods which only operate with :code:`ivy.NativeArray` instances, such as:code:`torch.Tensor` etc.

.. code-block:: python

    x = ivy.array([0., 1., 2.])
    y = ivy.reshape(x, (1, 3, 1))
    # passes without error


**Round Up**

That should hopefully be enough to get you started with the Ivy Array 😊

Please check out the discussions on the `repo <https://github.com/unifyai/ivy>`_ for FAQs, and reach out on `discord <https://discord.gg/ZVQdvbzNQJ>`_ if you have any questions!
