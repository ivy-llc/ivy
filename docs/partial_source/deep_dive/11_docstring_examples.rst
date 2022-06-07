Docstring Examples
==================

.. _`docstring examples discussion`: https://github.com/unifyai/ivy/discussions/1322
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`docstring examples channel`: https://discord.com/channels/799879767196958751/982738352103129098

After writing the general docstring, the final step is to add helpful examples to the docstring.

There are three types of examples. There are *functional* examples, which show the function being called like so
:code:`ivy.func_name(...)`, there are *instance method* examples,
show the method being called like so :code:`x.func_name(...)`,
and there are also *operator* examples,
which show an operation being performed like so :code:`x + y`.

The first two example types are very common, while the latter is, unsurprisingly,
only relevant for *operator* functions
such as :code:`ivy.add`, `ivy.subtract`, :code:`ivy.multiply` and :code:`ivy.divide`.

**Functional Examples**

Firstly, we should include *functional* examples with :code:`ivy.Array` instances in the input.

These should:

1. cover all possible variants (explained below) for each of the arguments independently,
   not combinatorily. This means the number of examples should be equal to the maximum number of
   variations for a single argument, and not the entire grid of variations across all arguments
   (further explained in the examples below)
2. vary the values and input shapes considerably between examples
3. start with the simplest examples first. For example, this means using the default values for all optional arguments
   in the first example, and using small arrays, with a small number of dimensions, and with *simple* values for the
   function in question
4. show an example with: (a) :code:`out` unused, (b) :code:`out` used to update a new array :code:`y`,
   and (c) :code:`out` used to inplace update the input array :code:`x`
5. If broadcasting is relevant for the function, then show examples which highlight this.
   For example, passing in different shapes for two array arguments

For all remaining examples, we can repeat input values from these :code:`ivy.Array` *functional*
examples covered by points 1-5.

The purpose of the extra examples with different input types in points 6-14 is to highlight the versatility
of the function. The purpose is not to provide an excessive number of variations of possible function inputs.

Next, we add examples with :code:`ivy.NativeArray` instances in the input.

There should be an example that:

6. passes in :code:`ivy.NativeArray` instances for all array arguments

For functions which accept more than one array, there should also be an example that:

7. passes in a combination of :code:`ivy.NativeArray` and :code:`ivy.Array` instances

Going further, for *nestable* functions there should be an example that:

8. passes in an :code:`ivy.Container` instance in place of one of the arguments

For *nestable* functions which accept more than one argument, there should also be an example that:

9. passes in :code:`ivy.Container` instances for multiple arguments

**Instance Method Examples**

*Instance method* examples are of course only relevant if an instance method for the function exists.
If so, these examples should:

10. call this instance method of the :code:`ivy.Array` class
11. call this instance method of the :code:`ivy.Container` class

**Operator Examples**

For example, calling any of (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.) on the array will result in
(:code:`__add__`, :code:`__sub__`, :code:`__mul__`, :code:`__truediv__` etc.) being called on the array class.

*Operator* examples are only relevant for *operator* functions. These are functions which are called when a
corresponding operator is applied to an array. For example, the functions :code:`ivy.add`, `ivy.subtract`,
:code:`ivy.multiply` and :code:`ivy.divide` are called when the operators :code:`+`, :code:`-`, :code:`*` and :code:`/`
are used respectively. Under the hood, these operators first call the special methods :code:`__add__`, :code:`__sub__`,
:code:`__mul__` and :code:`__truediv__` respectively, on either the :code:`ivy.Array` or :code:`ivy.Container`
instance upon which the operator is being applied.
These special methods in turn call the functions in the Ivy API mentioned above.

If the function is an *operator* function, then examples should be added which:

12. call the operator on :code:`ivy.Array` instances
13. call the operator on :code:`ivy.Container` instances
14. call the operator on a combination of :code:`ivy.Container` and :code:`ivy.Array` instances

**All Possible Variants**

Point 1 mentions that the examples should cover *all possible variations*.
Let’s look at an example to make it more clear what is meant by *all possible variants* of each argument independently.

Let’s take an imaginary function with the following argument spec:

.. code-block:: python

    def my_func(x: array,
                mode: Union[std, prod, var],
                some_flag: bool,
                another_flag: Optional[bool] = False,
                axes: Optional[Union[int, List[int]]]=-1):

In this case, our examples would need to include

*  :code:`x` being an :code:`array`
*  :code:`mode` being all of: :code:`std`, :code:`prod`, :code:`var`
*  :code:`some_flag` being both of: :code:`True`, :code:`False`
*  :code:`another_flag` being all of: :code:`default`, :code:`True`, :code:`False`
*  :code:`axis` being all of: :code:`default`, :code:`list`, :code:`int`.

Please note, this does not need to be done with a grid search.
There are 1 x 3 x 2 x 3 x 3 = 54 possible variations here, and we do not need an example for each one!
Instead, we only need as many examples as there are variations for the argument with the maximum number of variations,
in this case jointly being the :code:`mode`, :code:`another_flag` and :code:`axis` arguments, each with 3 variations.

For example, we could have three examples using the following arguments:

.. code-block:: python

    my_func(x0, std, True)
    my_func(x1, prod, False, True, [0, 1, 2])
    my_func(x2, var, True, False, 1)

It doesn’t matter how the variations are combined for the examples, as long as every variation for every argument is
included in the examples. These three examples procedurally go through the variations from left to right for each
argument, but this doesn’t need to be the case if you think other combinations make more sense for the examples.

You can also add more examples if you think some important use cases are missed, this is just a lower limit on the
examples that should be included in the docstring!

We'll next go through some examples to make these 14 points more clear.

ivy.tan
-------

The signature for :code:`ivy.tan` is as follows:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:

Let's start with the functional examples, with :code:`ivy.Array` instances in the input:

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

    >>> x = ivy.array([[1.1, 2.2, 3.3], \
                       [-4.4, -5.5, -6.6]])
    >>> ivy.tan(x, out=x)
    >>> print(x)
    ivy.array([[ 1.9647598, -1.3738229,  0.1597457],
               [-3.0963247,  0.9955841, -0.3278579]])

These examples cover points 1, 2, 3, 4 and 5.

Please note that in the above case of `x` having multi-line input, it is necessary for each line of the input
to be seperated by a '\\' so that they can be parsed by the script that tests the examples in the docstrings. 

Point 1 is simple to satisfy. Ignoring the union over :code:`ivy.Array` and :code:`ivy.NativeArray` which is covered by
points 6 and 7, and ignoring the *nestable* nature of the function which is covered by points 8 and 9,
then as far as point 1 is concerned, the input :code:`x` only has one possible variation. It must be an array.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, there are no optional inputs (aside from :code:`out`) and so this point is irrelevant,
and the values and shapes do become increasingly *complex*.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as
explained in point 4.

Point 5 is not relevant, as there is only one array input, and so broadcasting rules do not apply.

We then also add an example with an :code:`ivy.NativeArray` input, in order to satisfy point 6.
Point 7 is not relevant as there is only one array input
(excluding :code:`out` which does not count, as it essentially acts as an output)

.. code-block:: python

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0, 1, 2])
    >>> y = ivy.tan(x)
    >>> print(y)
    ivy.array([0., 1.5574077, -2.1850398])

We then also add an example with an :code:`ivy.Container` input, in order to satisfy point 8.
Point 9 is not relevant as there is only one input argument
(excluding :code:`out` which does not count, as it essentially acts as an output)

.. code-block:: python

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.tan(x)
    >>> print(y)
    {
        a: ivy.array([0., 1.5574077, -2.1850398]),
        b: ivy.array([-0.14254655, 1.1578213, -3.380515])
    }

We then add instance method examples to satisfy points 10 and 11.

.. code-block:: python

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.tan()
    >>> print(y)
    ivy.array([0., 1.5574077, -2.1850398])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.tan()
    >>> print(y)
    {
        a: ivy.array([0., 1.5574077, -2.1850398]),
        b: ivy.array([-0.14254655, 1.1578213, -3.380515])
    }

Points 12-14 are not relevant, as :code:`ivy.tan` is not an *operator* function.

ivy.roll
--------

The signature for :code:`ivy.roll` is as follows:

.. code-block:: python

    def roll(
        x: Union[ivy.Array, ivy.NativeArray],
        shift: Union[int, Sequence[int]],
        axis: Optional[Union[int, Sequence[int]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

Let's start with the functional examples, with :code:`ivy.Array` instances in the input:

.. code-block:: python

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    ivy.array([2., 0., 1.])

    >>> x = ivy.array([[0., 1., 2.], \
                       [3., 4., 5.]])
    >>> y = ivy.zeros((2, 3))
    >>> ivy.roll(x, 2, -1, out=y)
    >>> print(y)
    ivy.array([[1., 2., 0.],
               [4., 5., 3.]])

    >>> x = ivy.array([[[0., 0.], [1., 3.], [2., 6.]], \
                       [[3., 9.], [4., 12.], [5., 15.]]])
    >>> ivy.roll(x, (1, -1), (0, 2), out=x)
    >>> print(x)
    ivy.array([[[ 9., 3.],
                [12., 4.],
                [15., 5.]],
               [[ 0., 0.],
                [ 3., 1.],
                [ 6., 2.]]])

These examples cover points 1, 2, 3, 4 and 5.

Again, please note that in the above case of `x` having multi-line input, it is necessary for each line of the input
to be seperated by a '\\' so that they can be parsed by the script that tests the examples in the docstrings.

Point 1 is a bit less trivial to satisfy than it was for :code:`ivy.tan` above. While :code:`x` again only has one
variation (for the same reason as explained in the :code:`ivy.tan` example above), :code:`shift` has two variations
(:code:`int` or sequence of :code:`int`), and :code:`axis` has three variations
(:code:`int`, :sequence of :code:`int`, or :code:`None`).

Therefore, we need at least three examples (equal to the maximum number of variations, in this case :code:`axis`),
in order to show all variations for each argument. By going through each of the three examples above, it can be seen
that each variation for each argument is demonstrated in at least one of the examples. Therefore, point 1 is satisfied.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, as the first example uses the default values for optional arguments,
and the subsequent examples the non-default values in increasingly *complex* examples.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as
explained in point 4.

Point 5 is not relevant, as there is only one array input, and so broadcasting rules do not apply.

We then also add an example with an :code:`ivy.NativeArray` input, in order to satisfy point 6.
Point 7 is not relevant as there is only one array input
(excluding :code:`out` which does not count, as it essentially acts as an output)

.. code-block:: python

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    ivy.array([2., 0., 1.])

We then also add an example with an :code:`ivy.Container` for one of the inputs, in order to satisfy point 8.

.. code-block:: python

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }

Unlike :code:`ivy.tan`, point 9 is relevant in this case,
as there are three function inputs in total (excluding :code:`out`).
We can therefore add an example with multiple :code:`ivy.Container` inputs,
in order to satisfy point 9.

.. code-block:: python

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> shift = ivy.Container(a=1, b=-1)
    >>> y = ivy.roll(x, shift)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([4., 5., 3.])
    }

We then add instance method examples to satisfy points 10 and 11.

.. code-block:: python

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.roll(1)
    >>> print(y)
    ivy.array([2., 0., 1.])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.roll(1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.], dtype=float32),
        b: ivy.array([5., 3., 4.], dtype=float32)
    }

Points 12-14 are not relevant, as :code:`ivy.roll` is not an *operator* function.

ivy.add
-------

The signature for :code:`ivy.add` is as follows:

.. code-block:: python

    def add(
        x1: Union[ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:

Let's start with the functional examples, with :code:`ivy.Array` instances in the input:

.. code-block:: python

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

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

Again, please note that in the above case of `x` having multi-line input, it is necessary for each line of the input
to be seperated by a '\\' so that they can be parsed by the script that tests the examples in the docstrings.

Point 1 is again trivial to satisfy, as was the case for :code:`ivy.tan`.
Ignoring the union over :code:`ivy.Array` and :code:`ivy.NativeArray` which is covered by points 6 and 7,
and also ignoring the *nestable* nature of the function which is covered by points 8 and 9,
then as far as point 1 is concerned, inputs :code:`x1` and :code:`x2` both only have one possible variation.
They must both be arrays.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, there are no optional inputs (aside from :code:`out`) and so this point is irrelevant,
and the values and shapes do become increasingly *complex*.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as
explained in point 4.

Point 5 is satisfied, as the second example uses different shapes for the inputs :code:`x1` and :code:`x2`. This causes
the broadcasting rules to apply, which dictates how the operation is performed and the resultant shape of the output.

We then also add an example with an :code:`ivy.NativeArray` input, in order to satisfy point 6.

.. code-block:: python

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1, 2, 3])
    >>> y = ivy.native_array([4, 5, 6])
    >>> z = ivy.add(x, y)
    >>> print(z)
    ivy.array([5, 7, 9])

Unlike :code:`ivy.tan` and :code:`ivy.roll` above, point 7 is relevant in this case, as there are two array inputs.
We also add an example with a mix of :code:`ivy.NativeArray` and :code:`ivy.Array` inputs, in order to satisfy point 7.

.. code-block:: python

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.native_array([4, 5, 6])
    >>> z = ivy.add(x, y)
    >>> print(z)
    ivy.array([5, 7, 9])

We then also add an example with an :code:`ivy.Container` for one of the inputs, in order to satisfy point 8.

.. code-block:: python

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\
                          b=ivy.array([[5.], [6.], [7.]]))
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

Again, unlike :code:`ivy.tan`, point 9 is relevant in this case,
as there are two function inputs in total (exluding :code:`out`).
We can therefore add an example with multiple :code:`ivy.Container` inputs,
in order to satisfy point 9.

.. code-block:: python

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                          b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\
                          b=ivy.array([5, 6, 7]))
    >>> z = ivy.add(x, y)
    >>> print(z)
    {
        a: ivy.array([5, 7, 9]),
        b: ivy.array([7, 9, 11])
    }

We then add instance method examples to satisfy points 10 and 11.

.. code-block:: python

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = x.add(y)
    >>> print(z)
    ivy.array([5, 7, 9])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\ 
                          b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\ 
                          b=ivy.array([5, 6, 7]))
    >>> z = x.add(y)
    >>> print(z)
    {
        a: ivy.array([5, 7, 9]),
        b: ivy.array([7, 9, 11])
    }

Finally, we consider points 12-14. For :code:`ivy.tan` and :code:`ivy.roll` these were not relevant.
However, :code:`ivy.add` *is* an operator function. We therefore add the three operator examples which satisfy points
12, 13 and 14 respectively.

.. code-block:: python

    Operator Examples
    -----------------

    With :code:`ivy.Array` instances:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> z = x + y
    >>> print(z)
    ivy.array([5, 7, 9])

    With :code:`ivy.Container` instances:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\ 
                          b=ivy.array([2, 3, 4]))
    >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\ 
                          b=ivy.array([5, 6, 7]))
    >>> z = x + y
    >>> print(z)
    {
        a: ivy.array([5, 7, 9]),
        b: ivy.array([7, 9, 11])
    }

    With mix of :code:`ivy.Array` and :code:`ivy.Container` instances:

    >>> x = ivy.array([[1.1, 2.3, -3.6]])
    >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\ 
                          b=ivy.array([[5.], [6.], [7.]]))
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

**Round Up**

These three examples should give you a good understanding of what is required when adding docsting examples.

If you're ever unsure of how best to proceed,
please feel free to engage with the `docstring examples discussion`_,
or reach out on `discord`_ in the `docstring examples channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/rtce8XthiKA">
    </iframe>