``ivy.unify()``
================

..

   ⚠️ **Warning**: The compiler and the transpiler are not publicly available yet, so certain parts of this doc won't work as expected as of now!

Ivy's Unify function is an alias for ``ivy.transpile(..., to="ivy", ...)``. You can know
more about the transpiler in the `transpile() <transpile.rst>`_ page.

Unify API
---------

.. py:function:: ivy.unify(*objs, source = None, args = None, kwargs = None, **transpile_kwargs,)

  Transpiles an object into Ivy code. It's an alias to
  ``ivy.transpile(..., to="ivy", ...)``

  :param objs: Native callable(s) to transpile.
  :type objs: ``Callable``
  :param source: The framework that ``obj`` is from. This must be provided unless ``obj`` is a framework-specific module.
  :type source: ``Optional[str]``
  :param args: If specified, arguments that will be used to unify eagerly.
  :type args: ``Optional[Tuple]``
  :param kwargs: If specified, keyword arguments that will be used to unify eagerly.
  :type kwargs: ``Optional[dict]``
  :param transpile_kwargs: Arbitrary keyword arguments that will be passed to ``ivy.transpile``.

  :rtype: ``Union[Graph, LazyGraph, ModuleType, ivy.Module]``
  :return: A transpiled ``Graph`` or a non-initialized ``LazyGraph``. If the object is a native trainable module, the corresponding module in the target framework will be returned. If the object is a ``ModuleType``, the function will return a copy of the module with every method lazily transpiled.

Usage
-----

As we mentioned, ``ivy.unify()`` is an alias for ``ivy.transpile(..., to="ivy", ...)``.
So you can ues it in the same way as ``ivy.transpile()``.

.. code-block:: python

  import ivy
  ivy.set_backend("jax")

  def test_fn(x):
      return jax.numpy.sum(x)

  x1 = ivy.array([1., 2.])

  # transpiled_func and unified_func will have the same result
  transpiled_func = ivy.transpile(test_fn, to="ivy", args=(x1,))
  unified_func = ivy.unify(test_fn, args=(x1,))

Sharp bits
----------

``ivy.unify()`` has the same sharp bits as ``ivy.transpile()``. You can know more about
them in the :ref:`overview/one_liners/transpile:Sharp bits` section of the transpiler.

.. TODO add more examples explaining how unify is different from transpile