Docstring Examples
==================

The final step is to add helpful examples to the docstring.

There are two types of examples. There are *functional* examples, which show the function being called like so
:code:`ivy.func_name(...)`

There are also *instance method* examples which are called like so :code:`x.func_name(...)`

**Functional Examples**

The *functional* examples should:

1. cover all possible variants for each of the arguments independently, not combinatorily. This means the number of examples should be equal to the maximum number of variations for a single argument, and not the entire grid of variations across all arguments (further explained in the examples below)
2. vary the values and input shapes considerably between examples
3. start with the simplest examples first. For example, this means using the default values for all optional arguments in the first example, and using small arrays, with a small number of dimensions, and with *simple* values for the function in question.
4. show an example with: (a) :code:`out` unused, (b) :code:`out` used to update a new array :code:`y`, and (c) :code:`out` used to inplace update the input array :code:`x`

For *flexible* functions, there should also be an example that:

5. passes in :code:`ivy.Container` instances in place of all array arguments

For *flexible* functions which accept more than one array, there should also be an example that:

6. passes in a combination of :code:`ivy.Container` and :code:`ivy.Array` instances

**Instance Method Examples**

*Instance method* examples are only relevant for *flexible* functions. These examples should:

7. call the function as an instance method on the :code:`ivy.Array` class
8. call the function as an instance method on the :code:`ivy.Container` class

ivy.tan
-------

The signature for :code:`ivy.tan` is as follows:

.. code-block:: python

    def tan(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None
    ) -> Union[ivy.Array, ivy.Container]:

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

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    >>>                [-4.4, -5.5, -6.6]])
    >>> ivy.tan(x, out=x)
    >>> print(x)
    ivy.array([[ 1.9647598, -1.3738229,  0.1597457],
               [-3.0963247,  0.9955841, -0.3278579]])

These examples cover points 1, 2, 3 and 4.

Point 1 is simple to satisfy. Ignoring :code:`ivy.Container` which is covered by points 4 and 5,
and ignoring the union over :code:`ivy.Array` and :code:`ivy.NativeArray` which is true for **all** array inputs,
then as far as point 1 is concerned, the input :code:`x` only has one possible variation. It must be an array.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, there are no optional inputs (aside from :code:`out`) and so this point is irrelevant, and the values and shapes do become increasingly *complex*.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as
explained in point 4.

We then also add an example with an :code:`ivy.Container` input, in order to satisfy point 5.
Point 6 is not relevant as there is only one array input
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

Finally, we add instance method examples to satisfy points 7 and 8.

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

ivy.roll
--------

The signature for :code:`ivy.roll` is as follows:

.. code-block:: python

    def roll(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        shift: Union[int, Tuple[int, ...]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:

Let's start with the functional examples, with :code:`ivy.Array` instances in the input:

.. code-block:: python

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    ivy.array([2., 0., 1.])

    >>> x = ivy.array([[0., 1., 2.],
    >>>                [3., 4., 5.]])
    >>> y = ivy.zeros((2, 3))
    >>> ivy.roll(x, 2, -1, out=y)
    >>> print(y)
    ivy.array([[1., 2., 0.],
               [4., 5., 3.]])

    >>> x = ivy.array([[[0., 0.], [1., 3.], [2., 6.]],
    >>>                [[3., 9.], [4., 12.], [5., 15.]]])
    >>> ivy.roll(x, (1, -1), (0, 2), out=x)
    >>> print(x)
    ivy.array([[[ 9., 3.],
                [12., 4.],
                [15., 5.]],
               [[ 0., 0.],
                [ 3., 1.],
                [ 6., 2.]]])

These examples cover points 1, 2, 3 and 4.

Point 1 is a bit less trivial to satisfy than it was for :code:`ivy.tan` above. While :code:`x` again only has one
variation (for the same reason as explained in the :code:`ivy.tan` example above), :code:`shift` has two variations
(:code:`int` or :code:`tuple` of :code:`int`), and :code:`axis` has three variations
(:code:`int`, :code:`tuple` of :code:`int`, or :code:`None`).

Therefore, we need at least three examples (equal to the maximum number of variations, in this case :code:`axis`),
in order to show all variations for each argument. By going through each of the three examples above, it can be seen
that each variation for each argument is demonstrated in at least one of the examples. Therefore, point 1 is satisfied.

Point 2 is satisfied, as the shape and values of the inputs are varied between each of the three examples.

Point 3 is satisfied, as the first example uses the default values for optional arguments, and the subsequent examples the non-default values in increasingly *complex* examples.

Point 4 is clearly satisfied, as each of the three examples shown above use the :code:`out` argument exactly as
explained in point 4.

We then also add an example with an :code:`ivy.Container` input, in order to satisfy point 5.
Point 6 is not relevant as there is again only one array input
(excluding :code:`out` which does not count, as it essentially acts as an output).

.. code-block:: python

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    >>>                   b=ivy.array([3., 4., 5.]))
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }

Finally, we add instance method examples to satisfy points 7 and 8.

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