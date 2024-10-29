``ivy.transpile()``
===================

Ivy's Transpiler converts a function, class, or module written in any framework into
your framework of choice, preserving all the logic between frameworks.


Transpiler API
--------------

.. py:function:: ivy.transpile(object, source="torch", target="tensorflow", profiling=False,)

  Converts a given object (class/function) from one framework to another.

  :param object: The object (class/function) to be transpiled to a new framework.
  :type object: ``Any``
  :param source: The source framework we're converting the object from.
  :type source: ``Optional[str]``
  :param target: The target framework we're converting the object to.
  :type target: ``Optional[str]``
  :param profiling: Whether to use performance profiling.
  :type profiling: ``bool``
  :rtype: ``Union[MethodType, FunctionType, type]``
  :return: The transpiled object.
