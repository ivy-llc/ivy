Docstring Examples
==================

.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`docstring examples channel`: https://discord.com/channels/799879767196958751/982738352103129098
.. _`docstring examples forum`: https://discord.com/channels/799879767196958751/1028297703089774705

After writing the general docstrings, the final step is to add helpful examples to the docstrings.

There are eight types of examples, which each need to be added:

**Functional** examples show the function being called like so :code:`ivy.func_name(...)`, and these should be added to docstring of the function in the Ivy API :func:`ivy.func_name`.

**Array instance method** examples show the method being called like so :code:`x.func_name(...)` on an :class:`ivy.Array` instance, and these should be added to the docstring of the :class:`ivy.Array` instance method :meth:`ivy.Array.func_name`.

**Container instance method** examples show the method being called like so :code:`x.func_name(...)` on an :class:`ivy.Container` instance, and these should be added to the docstring of the :class:`ivy.Container` instance method :meth:`ivy.Container.func_name`.

**Array operator** examples show an operation being performed like so :code:`x + y` with :code:`x` being an :class:`ivy.Array` instance, and these should be added to the docstring of the :class:`ivy.Array` special method :meth:`ivy.Array.__<op_name>__`.

**Array reverse operator** examples show an operation being performed like so :code:`x + y` with :code:`x` being a :code:`Number` and :code:`y` being an :class:`ivy.Array` instance. These should be added to the docstring of the :class:`ivy.Array` reverse special method :meth:`ivy.Array.__r<op_name>__`.

**Container operator** examples show an operation being performed like so :code:`x + y` with :code:`x` being an :class:`ivy.Container` instance, and these should be added to the docstring of the :class:`ivy.Container` special method :meth:`ivy.Container.__<op_name>__`.

**Container reverse operator** examples show an operation being performed like so :code:`x + y` with :code:`x` being a :code:`Number` and :code:`y` being an :class:`ivy.Container` instance. These should be added to the docstring of the :class:`ivy.Container` reverse special method :meth:`ivy.Container.__r<op_name>__`.

The first three example types are very common, while the last four, unsurprisingly, are only relevant for *operator* functions such as :func:`ivy.add`, :func:`ivy.subtract`, :func:`ivy.multiply` and :func:`ivy.divide`.

For example, calling any of (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.) on the array will result in (:meth:`__add__`, :meth:`__sub__`, :meth:`__mul__`, :meth:`__truediv__` etc.) being called on the array class.

**Operator** examples are only relevant for *operator* functions. These are functions which are called when a corresponding operator is applied to an array.
For example, the functions :func:`ivy.add`, :func:`ivy.subtract`, :func:`ivy.multiply` and :func:`ivy.divide` are called when the operators :code:`+`, :code:`-`, :code:`*` and :code:`/` are used respectively.
Under the hood, these operators first call the special methods :meth:`__add__`, :meth:`__sub__`, :meth:`__mul__` and :meth:`__truediv__` respectively, on either the :class:`ivy.Array` or :class:`ivy.Container` instance upon which the operator is being applied.
These special methods in turn call the functions in the Ivy API mentioned above.

**Functional Examples**

To recap, *functional* examples show the function being called like so :code:`ivy.func_name(...)`, and these should be added to docstring of the function in the Ivy API :func:`ivy.func_name`.

Firstly, we should include *functional* examples with :class:`ivy.Array` instances in the input.

These should:

1. cover all possible variants (explained below) for each of the arguments independently, not combinatorially. This means the number of examples should be equal to the maximum number of variations for a single argument, and not the entire grid of variations across all arguments (further explained in the examples below)

2. vary the values and input shapes considerably between examples

3. start with the simplest examples first. For example, this means using the default values for all optional arguments in the first example, and using small arrays, with a small number of dimensions, and with *simple* values for the function in question

4. show an example with: (a) :code:`out` unused, (b) :code:`out` used to update a new array :code:`y`, and (c) :code:`out` used to inplace update the input array :code:`x` (provided that it shares the same :code:`dtype` and :code:`shape` as the return)

5. If broadcasting is relevant for the function, then show examples which highlight this.
   For example, passing in different shapes for two array arguments

For all remaining examples, we can repeat input values from these :class:`ivy.Array` *functional* examples covered by points 1-5.

The purpose of the extra examples with different input types in points 6-18 is to highlight the different contexts in which the function can be called (as an instance method etc.).
The purpose is not to provide an excessive number of variations of possible function inputs.

Next, for *nestable* functions there should be an example that:

6. passes in an :class:`ivy.Container` instance in place of one of the arguments

For *nestable* functions which accept more than one argument, there should also be an example that:

7. passes in :class:`ivy.Container` instances for multiple arguments

In all cases, the containers should have at least two leaves.
For example, the following container is okay to use for example purposes:

.. code-block:: python

    x = ivy.Container(a=ivy.array([0.]), b=ivy.array([1.]))

Whereas the following container is not okay to use for example purposes:

.. code-block:: python

    x = ivy.Container(a=ivy.array([0.]))

**Array Instance Method Example**

To recap, *array instance method* examples show the method being called like so :code:`x.func_name(...)` on an :class:`ivy.Array` instance, and these should be added to the docstring of the :class:`ivy.Array` instance method :meth:`ivy.Array.func_name`.

These examples are of course only relevant if an instance method for the function exists. If so, this example should simply:

8. call this instance method of the :class:`ivy.Array` class

**Container Instance Method Example**

To recap, *container instance method* examples show the method being called like so :code:`x.func_name(...)` on an :class:`ivy.Container` instance, and these should be added to the docstring of the :class:`ivy.Container` instance method :meth:`ivy.Container.func_name`.

These examples are of course only relevant if an instance method for the function exists.
If so, this example should simply:

9. call this instance method of the :class:`ivy.Container` class

**Array Operator Examples**

To recap, *array operator* examples show an operation being performed like so :code:`x + y` with :code:`x` being an :class:`ivy.Array` instance, and these should be added to the docstring of the :class:`ivy.Array` special method :meth:`ivy.Array.__<op_name>__`.

If the function is an *operator* function, then the *array operator* examples should:

10. call the operator on two :class:`ivy.Array` instances
11. call the operator with an :class:`ivy.Array` instance on the left and :class:`ivy.Container` on the right

**Array Reverse Operator Example**

To recap, *array reverse operator* examples show an operation being performed like so :code:`x + y` with :code:`x` being a :code:`Number` and :code:`y` being an :class:`ivy.Array` instance. These should be added to the docstring of the :class:`ivy.Array` reverse special method :meth:`ivy.Array.__r<op_name>__`.

If the function is an *operator* function, then the *array reverse operator* example should:

12. call the operator with a :code:`Number` on the left and an :class:`ivy.Array` instance on the right

**Container Operator Examples**

To recap, *container operator* examples show an operation being performed like so :code:`x + y` with :code:`x` being an :class:`ivy.Container` instance, and these should be added to the docstring of the :class:`ivy.Container` special method :meth:`ivy.Container.__<op_name>__`.

If the function is an *operator* function, then the *container operator* examples should:

13. call the operator on two :class:`ivy.Container` instances containing :code:`Number` instances at the leaves
14. call the operator on two :class:`ivy.Container` instances containing :class:`ivy.Array` instances at the leaves
15. call the operator with an :class:`ivy.Container` instance on the left and :class:`ivy.Array` on the right

**Container Reverse Operator Example**

To recap, *container reverse operator* examples show an operation being performed like so :code:`x + y` with :code:`x` being a :code:`Number` and :code:`y` being an :class:`ivy.Container` instance.
These should be added to the docstring of the :class:`ivy.Container` reverse special method :meth:`ivy.Container.__r<op_name>__`.

If the function is an *operator* function, then the *array reverse operator* example should:

16. call the operator with a :code:`Number` on the left and an :class:`ivy.Container` instance on the right

**Note**

All docstrings must run without error for all backend frameworks.
If some backends do not support some :code:`dtype` for a function, then we should not include this :code:`dtype` for any of the examples for that particular function in the docstring.

**All Possible Variants**

Point 1 mentions that the examples should cover *all possible variations*.
Let’s look at an example to make it more clear what is meant by *all possible variants* of each argument independently.

Let’s take an imaginary function with the following argument spec:

.. code-block:: python

    def my_func(x: array,
                mode: Union[std, prod, var],
                some_flag: bool,
                another_flag: bool = False,
                axes: Optional[Union[int, List[int]]]=-1):

In this case, our examples would need to include

*  :code:`x` being an :code:`array`
*  :code:`mode` being all of: :code:`std`, :code:`prod`, :code:`var`
*  :code:`some_flag` being both of: :code:`True`, :code:`False`
*  :code:`another_flag` being all of: :code:`default`, :code:`True`, :code:`False`
*  :code:`axis` being all of: :code:`default`, :code:`list`, :code:`int`.

Please note, this does not need to be done with a grid search.
There are 1 x 3 x 2 x 3 x 3 = 54 possible variations here, and we do not need an example for each one!
Instead, we only need as many examples as there are variations for the argument with the maximum number of variations, in this case jointly being the :code:`mode`, :code:`another_flag` and :code:`axis` arguments, each with 3 variations.

For example, we could have three examples using the following arguments:

.. code-block:: python

    my_func(x0, std, True)
    my_func(x1, prod, False, True, [0, 1, 2])
    my_func(x2, var, True, False, 1)

It doesn’t matter how the variations are combined for the examples, as long as every variation for every argument is included in the examples.
These three examples procedurally go through the variations from left to right for each argument, but this doesn’t need to be the case if you think other combinations make more sense for the examples.

You can also add more examples if you think some important use cases are missed, this is just a lower limit on the examples that should be included in the docstring!

We'll next go through some examples to make these 18 points more clear.

ivy.tan
-------

**Functional Examples**

The signature for :func:`ivy.tan` is as follows:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:

Let's start with the functional examples, with :class:`ivy.Array` instances in the input:

.. parsed-literal::

    Examples
    --------
    With :class:`ivy.Array` input:

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
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.tan(x, out=x)
    >>> print(x)
    ivy.array([[ 1.9647598, -1.3738229,  0.1597457],
               [-3.0963247,  0.9955841, -0.3278579]])

These examples cover points 1, 2, 3, 4 and 5.

Please note that in the above case of `x` having multi-line input, it is necessary for each line of the input to be seperated by a '...\' so that they can be parsed by the script that tests the examples in the docstrings.

Point 1 is simple to satisfy.
Ignoring the union over :class:`ivy.Array` and :class:`ivy.NativeArray` which is covered by points 6 and 7, and ignoring the *nestable* nature of the function which is covered by points 8 and 9, then as far as point 1 is concerned, the input :code:`x` only has one possible variation.
It must be an array.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, there are no optional inputs (aside from :code:`out`) and so this point is irrelevant, and the values and shapes do become increasingly *complex*.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as explained in point 4.
The return has the same :code:`shape` and :code:`dtype` as the input, making all three examples possible.

Point 5 is not relevant, as there is only one array input, and so broadcasting rules do not apply.

We then also add an example with an :class:`ivy.Container` input, in order to satisfy point 6.
Point 7 is not relevant as there is only one input argument (excluding :code:`out` which does not count, as it essentially acts as an output)

.. parsed-literal::

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.tan(x)
    >>> print(y)
    {
        a: ivy.array([0., 1.5574077, -2.1850398]),
        b: ivy.array([-0.14254655, 1.1578213, -3.380515])
    }

**Array Instance Method Example**

We then add an instance method example to :meth:`ivy.Array.tan` in order to satisfy
point 8.

.. code-block:: python

    Examples
    --------
    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.tan()
    >>> print(y)
    ivy.array([0., 1.56, -2.19])

**Container Instance Method Example**

We then add an instance method example to :meth:`ivy.Container.tan` in order to satisfy point 9.

.. code-block:: python

    Examples
    --------
    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.tan()
    >>> print(y)
    {
        a:ivy.array([0., 1.56, -2.19]),
        b:ivy.array([-0.143, 1.16, -3.38])
    }

**Array Operator Examples**

Points 10 and 11 are not relevant as :func:`ivy.tan` is not an *operator* function.

**Array Reverse Operator Example**

Point 12 is not relevant as :func:`ivy.tan` is not an *operator* function.

**Container Operator Examples**

Points 13, 14 and 15 are not relevant as :func:`ivy.tan` is not an *operator* function.

**Container Reverse Operator Example**

Point 16 is not relevant as :func:`ivy.tan` is not an *operator* function.

ivy.roll
--------

**Functional Examples**

The signature for :func:`ivy.roll` is as follows:

.. code-block:: python

    def roll(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

Let's start with the functional examples, with :class:`ivy.Array` instances in the input:

.. parsed-literal::

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    ivy.array([2., 0., 1.])

    >>> x = ivy.array([[0., 1., 2.],
    ...                [3., 4., 5.]])
    >>> y = ivy.zeros((2, 3))
    >>> ivy.roll(x, 2, -1, out=y)
    >>> print(y)
    ivy.array([[1., 2., 0.],
               [4., 5., 3.]])

    >>> x = ivy.array([[[0., 0.], [1., 3.], [2., 6.]],
    ...                 [[3., 9.], [4., 12.], [5., 15.]]])
    >>> ivy.roll(x, (1, -1), (0, 2), out=x)
    >>> print(x)
    ivy.array([[[ 9., 3.],
                [12., 4.],
                [15., 5.]],
               [[ 0., 0.],
                [ 3., 1.],
                [ 6., 2.]]])

These examples cover points 1, 2, 3, 4 and 5.

Again, please note that in the above case of `x` having multi-line input, it is necessary for each line of the input to be seperated by a '...\' so that they can be parsed by the script that tests the examples in the docstrings.

Point 1 is a bit less trivial to satisfy than it was for :func:`ivy.tan` above.
While :code:`x` again only has one variation (for the same reason as explained in the :func:`ivy.tan` example above), :code:`shift` has two variations (:code:`int` or sequence of :code:`int`), and :code:`axis` has three variations (:code:`int`, :sequence of :code:`int`, or :code:`None`).

Therefore, we need at least three examples (equal to the maximum number of variations, in this case :code:`axis`), in order to show all variations for each argument.
By going through each of the three examples above, it can be seen that each variation for each argument is demonstrated in at least one of the examples.
Therefore, point 1 is satisfied.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, as the first example uses the default values for optional arguments, and the subsequent examples the non-default values in increasingly *complex* examples.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as explained in point 4.
The return has the same :code:`shape` and :code:`dtype` as the input, making all three examples possible.

Point 5 is not relevant, as there is only one array input, and so broadcasting rules do not apply.

We then also add an example with an :class:`ivy.Container` for one of the inputs, in order to satisfy point 6.

.. parsed-literal::

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }

Unlike :func:`ivy.tan`, point 7 is relevant in this case, as there are three function inputs in total (excluding :code:`out`).
We can therefore add an example with multiple :class:`ivy.Container` inputs, in order to satisfy point 7.

.. parsed-literal::

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3., 4., 5.]))
    >>> shift = ivy.Container(a=1, b=-1)
    >>> y = ivy.roll(x, shift)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([4., 5., 3.])
    }

**Array Instance Method Example**

We then add an instance method example to :meth:`ivy.Array.roll` in order to satisfy point 8.

.. code-block:: python

    Examples
    --------
    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.roll(1)
    >>> print(y)
    ivy.array([2., 0., 1.])

**Container Instance Method Example**

We then add an instance method example to :meth:`ivy.Container.roll` in order to satisfy point 9.

.. code-block:: python

    Examples
    --------

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.roll(1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.], dtype=float32),
        b: ivy.array([5., 3., 4.], dtype=float32)
    }


**Array Operator Examples**

Points 10 and 11 are not relevant as :func:`ivy.roll` is not an *operator* function.

**Array Reverse Operator Example**

Point 12 is not relevant as :func:`ivy.roll` is not an *operator* function.

**Container Operator Examples**

Points 13, 14 and 15 are not relevant as :func:`ivy.roll` is not an *operator* function.

**Container Reverse Operator Example**

Point 16 is not relevant as :code:`func.roll` is not an *operator* function.

ivy.add
-------

**Functional Examples**

The signature for :func:`ivy.add` is as follows:

.. code-block:: python

    def add(
        x1: Union[ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

Let's start with the functional examples, with :class:`ivy.Array` instances in the input:

.. parsed-literal::

    Examples
    --------

    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = ivy.add(x, y)
    >>> print(z)
    ivy.array([5, 7, 9])

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> y = ivy.array([[4.8], [5.2], [6.1]])
    >>> z = ivy.zeros((3, 3))
    >>> ivy.add(x, y, out=z)
    >>> print(z)
    ivy.array([[5.9, 7.1, 1.2],
               [6.3, 7.5, 1.6],
               [7.2, 8.4, 2.5]])

    >>> x = ivy.array([[[1.1], [3.2], [-6.3]]])
    >>> y = ivy.array([[8.4], [2.5], [1.6]])
    >>> ivy.add(x, y, out=x)
    >>> print(x)
    ivy.array([[[9.5],
                [5.7],
                [-4.7]]])

These examples cover points 1, 2, 3, 4 and 5.

Again, please note that in the above case of `x` having multi-line input, it is necessary for each line of the input to be seperated by a '...\' so that they can be parsed by the script that tests the examples in the docstrings.

Point 1 is again trivial to satisfy, as was the case for :func:`ivy.tan`.
Ignoring the union over :class:`ivy.Array` and :class:`ivy.NativeArray` which is covered by points 6 and 7, and also ignoring the *nestable* nature of the function which is covered by points 8 and 9, then as far as point 1 is concerned, inputs :code:`x1` and :code:`x2` both only have one possible variation.
They must both be arrays.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, there are no optional inputs (aside from :code:`out`) and so this point is irrelevant, and the values and shapes do become increasingly *complex*.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as explained in point 4.
The return has the same :code:`shape` and :code:`dtype` as the input, making all three examples possible.

Point 5 is satisfied, as the second example uses different shapes for the inputs :code:`x1` and :code:`x2`.
This causes the broadcasting rules to apply, which dictates how the operation is performed and the resultant shape of the output.

We then also add an example with an :class:`ivy.Container` for one of the inputs, in order to satisfy point 6.

.. parsed-literal::

    With one :class:`ivy.Container` input:

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
    ...                   b=ivy.array([[5.], [6.], [7.]]))
    >>> z = ivy.add(x, y)
    >>> print(z)
    {
        a: ivy.array([[5.1, 6.3, 0.4],
                      [6.1, 7.3, 1.4],
                      [7.1, 8.3, 2.4]]),
        b: ivy.array([[6.1, 7.3, 1.4],
                      [7.1, 8.3, 2.4],
                      [8.1, 9.3, 3.4]])
    }

Again, unlike :func:`ivy.tan`, point 7 is relevant in this case, as there are two function inputs in total (exluding :code:`out`).
We can therefore add an example with multiple :class:`ivy.Container` inputs, in order to satisfy point 7.

.. parsed-literal::

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                   b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),
    ...                   b=ivy.array([5, 6, 7]))
    >>> z = ivy.add(x, y)
    >>> print(z)
    {
        a: ivy.array([5, 7, 9]),
        b: ivy.array([7, 9, 11])
    }

**Array Instance Method Example**

We then add an instance method example to :meth:`ivy.Array.add` in order to satisfy point 8.

.. code-block:: python

    Examples
    --------

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = x.add(y)
    >>> print(z)
    ivy.array([5, 7, 9])

**Container Instance Method Example**

We then add an instance method example to :meth:`ivy.Container.add` in order to satisfy point 9.

.. code-block:: python

    Examples
    --------

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                   b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),
    ...                   b=ivy.array([5, 6, 7]))
    >>> z = x.add(y)
    >>> print(z)
    {
        a: ivy.array([5, 7, 9]),
        b: ivy.array([7, 9, 11])
    }

**Array Operator Examples**

Point 10 is satisfied by the following example in the :meth:`ivy.Array.__add__` docstring, with the operator called on two :class:`ivy.Array` instances.

.. parsed-literal::

    Examples
    --------

    With :class:`ivy.Array` instances only:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = x + y
    >>> print(z)
    ivy.array([5, 7, 9])

Point 11 is satisfied by the following example in the :meth:`ivy.Array.__add__` docstring, with the operator called with an :class:`ivy.Array` instance on the left and :class:`ivy.Container` on the right.

.. parsed-literal::

    With mix of :class:`ivy.Array` and :class:`ivy.Container` instances:

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
    ...                   b=ivy.array([[5.], [6.], [7.]]))
    >>> z = x + y
    >>> print(z)
    {
        a: ivy.array([[5.1, 6.3, 0.4],
                      [6.1, 7.3, 1.4],
                      [7.1, 8.3, 2.4]]),
        b: ivy.array([[6.1, 7.3, 1.4],
                      [7.1, 8.3, 2.4],
                      [8.1, 9.3, 3.4]])
    }

**Array Reverse Operator Examples**

Point 12 is satisfied by the following example in the :meth:`ivy.Array.__radd__` docstring, with the operator called with a :code:`Number` on the left and an :class:`ivy.Array` instance on the right.

.. code-block:: python

    Examples
    --------

    >>> x = 1
    >>> y = ivy.array([4, 5, 6])
    >>> z = x + y
    >>> print(z)
    ivy.array([5, 6, 7])

**Container Operator Examples**

Point 13 is satisfied by the following example in the :meth:`ivy.Container.__add__` docstring, with the operator called on two :class:`ivy.Container` instances containing :code:`Number` instances at the leaves.

.. parsed-literal::

    Examples
    --------

    With :code:`Number` instances at the leaves:

    >>> x = ivy.Container(a=1, b=2)
    >>> y = ivy.Container(a=3, b=4)
    >>> z = x + y
    >>> print(z)
    {
        a: 4,
        b: 6
    }

Point 14 is satisfied by the following example in the :meth:`ivy.Container.__add__` docstring, with the operator called on two :class:`ivy.Container` instances containing :class:`ivy.Array` instances at the leaves.

.. parsed-literal::

    With :class:`ivy.Array` instances at the leaves:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                   b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),
    ...                   b=ivy.array([5, 6, 7]))
    >>> z = x + y
    >>> print(z)
    {
        a: ivy.array([5, 7, 9]),
        b: ivy.array([7, 9, 11])
    }

Point 15 is satisfied by the following example in the :meth:`ivy.Container.__add__` docstring, with the operator called with an :class:`ivy.Container` instance on the left and :class:`ivy.Array` on the right.

.. parsed-literal::

    With a mix of :class:`ivy.Container` and :class:`ivy.Array` instances:

    >>> x = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
    ...                   b=ivy.array([[5.], [6.], [7.]]))
    >>> y = ivy.array([[1.1, 2.3, -3.6]])
    >>> z = x + y
    >>> print(z)
    {
        a: ivy.array([[5.1, 6.3, 0.4],
                      [6.1, 7.3, 1.4],
                      [7.1, 8.3, 2.4]]),
        b: ivy.array([[6.1, 7.3, 1.4],
                      [7.1, 8.3, 2.4],
                      [8.1, 9.3, 3.4]])
    }

**Container Reverse Operator Example**

Point 16 is satisfied by the following example in the :meth:`ivy.Container.__radd__` docstring, with the operator called with a :code:`Number` on the left and an :class:`ivy.Container` instance on the right.

.. code-block:: python

    Examples
    --------

    >>> x = 1
    >>> y = ivy.Container(a=3, b=4)
    >>> z = x + y
    >>> print(z)
    {
        a: 4,
        b: 5
    }
**Docstring Tests**

After making a Pull Request, each time you make a commit, then a number of checks are run on it to ensure everything's working fine.
One of these checks is the docstring tests named as :code:`test-docstrings / run-docstring-tests` in the GitHub actions.
The docstring tests check whether the docstring examples for a given function are valid or not.
It basically checks if the output upon execution of the examples that are documented match exactly with the ones shown in the docstrings.
Therefore each time you make a commit, you must ensure that the :code:`test-docstrings / run-docstring-tests` are working correctly at least for the function you are making changes to.
To check whether the docstring tests are passing you need to check the logs for :code:`test-docstrings / run-docstring-tests`:

    .. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/deep_dive/docstring_examples/docstring_failing_test_logs.png?raw=true
           :width: 420

You will need to go through the logs and see if the list of functions for which the docstring tests are failing also has the function you are working with.

If the docstring tests are failing the  logs show a list of functions having issues along with a diff message:
:code:`output for failing_fn_name on run: ......`
:code:`output in docs: ........`
as shown below:

    .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/deep_dive/docstring_examples/docstring_log.png
           :width: 420

It can be quite tedious to go through the output diffs and spot the exact error, so you can take help of online tools like `text compare <https://text-compare.com/>`_ to spot the minutest of differences.

Once you make the necessary changes and the function you are working on doesn't cause the docstring tests to fail, you should be good to go.
However, one of the reviewers might ask you to make additional changes involving examples.
Passing docstring tests is a necessary but not sufficient condition for the completion of docstring formatting.

.. note::
   Docstring examples should not have code that imports ivy or sets a backend, otherwise it leads to segmentation faults.

**Round Up**

These three examples should give you a good understanding of what is required when adding docstring examples.

If you have any questions, please feel free to reach out on `discord`_ in the `docstring examples channel`_ or in the `docstring examples forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/rtce8XthiKA" class="video">
    </iframe>
