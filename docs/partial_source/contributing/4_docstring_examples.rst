Docstring Examples
==================

The final step is to add helpful examples to the docstring. These should show *functional* examples that:

* cover all possible variants for each of the arguments
* show an example with: (a) :code:`out` unused, (b) :code:`out` used to update a new array :code:`y`, and (c) :code:`out` used to inplace update the input array :code:`x`
* vary the values and input shapes considerably between examples

For *flexible* functions, there should also be examples that:

* pass in :code:`ivy.Container` instances in place of all array arguments, in a *functional* example
* call the function as an instance method on the :code:`ivy.Array` class
* call the function as an instance method on the :code:`ivy.Container` class

For *flexible* functions which accept more than one array argument, there should also be an example that:

* passes in a combination of :code:`ivy.Container` and :code:`ivy.Array` instances, in a functional example

Let's start with the functional :code:`ivy.Array` examples for :code:`ivy.tan`:

.. code-block:: python

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 2])
    >>> y = ivy.tan(x)
    >>> print(y)
    ivy.array([0., 1.5574077, -2.1850398])

    >>> x = ivy.array([0.5, -0.7, 2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.tan(x, out=y)
    >>> print(y)
    ivy.array([0.5463025, -0.8422884, -0.91601413])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    >>>                [-4.4, -5.5, -6.6]])
    >>> ivy.tan(x, out=x)
    >>> print(x)
    ivy.array([[ 1.9647598, -1.3738229,  0.1597457],
               [-3.0963247,  0.9955841, -0.3278579]])