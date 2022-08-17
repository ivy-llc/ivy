Ivy Tests
=========

.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`hypothesis`: https://hypothesis.readthedocs.io/en/latest/
.. _`test_array_api`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/ivy_tests
.. _`test_ivy`: https://github.com/unifyai/ivy/tree/0fc4a104e19266fb4a65f5ec52308ff816e85d78/ivy_tests/test_ivy
.. _`commit`: https://github.com/unifyai/ivy/commit/8e6074419c0b6ee27c52e8563374373c8bcff30f
.. _`uploading`: https://github.com/unifyai/ivy/blob/0fc4a104e19266fb4a65f5ec52308ff816e85d78/.github/workflows/test-array-api-torch.yml#L30
.. _`downloading`: https://github.com/unifyai/ivy/blob/0fc4a104e19266fb4a65f5ec52308ff816e85d78/.github/workflows/test-array-api-torch.yml#L14
.. _`continuous integration`: https://github.com/unifyai/ivy/tree/0fc4a104e19266fb4a65f5ec52308ff816e85d78/.github/workflows
.. _`search strategies`: https://hypothesis.readthedocs.io/en/latest/data.html
.. _`methods`: https://hypothesis.readthedocs.io/en/latest/data.html
.. _`finfo`: https://github.com/unifyai/ivy/blob/d8f1ffe8ebf38fa75161c1a9459170e95f3c82b6/ivy/functional/ivy/data_type.py#L276
.. _`data generation`: https://github.com/unifyai/ivy/blob/7063bf4475b93f87a4a96ef26c56c2bd309a2338/ivy_tests/test_ivy/test_functional/test_core/test_dtype.py#L337
.. _`here`: https://lets-unify.ai/ivy/deep_dive/1_function_types.html#function-types
.. _`test_default_int_dtype`: https://github.com/unifyai/ivy/blob/7063bf4475b93f87a4a96ef26c56c2bd309a2338/ivy_tests/test_ivy/test_functional/test_core/test_dtype.py#L835
.. _`sampled_from`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.sampled_from
.. _`lists`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.lists
.. _`booleans`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.booleans
.. _`integers`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.integers
.. _`floats`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.floats
.. _`none`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.none
.. _`tuples`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.tuples
.. _`one_of`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.one_of
.. _`shared`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.shared
.. _`sets`: https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.sets
.. _`map`: https://hypothesis.readthedocs.io/en/latest/data.html#mapping
.. _`filter`: https://hypothesis.readthedocs.io/en/latest/data.html#filtering
.. _`flatmap`: https://hypothesis.readthedocs.io/en/latest/data.html#chaining-strategies-together
.. _`data`: https://hypothesis.readthedocs.io/en/latest/data.html?highlight=strategies.data#hypothesis.strategies.data
.. _`composite`: https://hypothesis.readthedocs.io/en/latest/data.html?highlight=strategies.composite#hypothesis.strategies.composite
.. _`line`: https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py#L477
.. _`here`: https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py#L392
.. _`this`: https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/test_functional/test_core/test_sorting.py#L18
.. _`example`: https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/helpers.py#L1085
.. _`test_concat`: https://github.com/unifyai/ivy/blob/1281a2baa15b8e43a06df8926ceef1a3d7605ea6/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py#L51
.. _`test_device`: https://github.com/unifyai/ivy/blob/master/ivy_tests/test_ivy/test_functional/test_core/test_device.py
.. _`test_manipulation`: https://github.com/unifyai/ivy/blob/master/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py
.. _`test_layers`: https://github.com/unifyai/ivy/blob/master/ivy_tests/test_ivy/test_functional/test_nn/test_layers.py
.. _`keyword`:https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/helpers.py#L1108
.. _`arguments`: https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/helpers.py#L1354
.. _`documentation`: https://hypothesis.readthedocs.io/en/latest/quickstart.html
.. _`test_gelu`: https://github.com/unifyai/ivy/blob/b2305d1d01528c4a6fa9643dfccf65e33b8ecfd8/ivy_tests/test_ivy/test_functional/test_nn/test_activations.py#L104
.. _`test_array_function`: https://github.com/unifyai/ivy/blob/0fc4a104e19266fb4a65f5ec52308ff816e85d78/ivy_tests/test_ivy/helpers.py#L401
.. _`artifact`: https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts
.. _`ivy tests discussion`: https://github.com/unifyai/ivy/discussions/1304
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`ivy tests channel`: https://discord.com/channels/799879767196958751/982738436383445073

On top of the Array API `test suite`_, which is included as a submodule mapped to the folder :code:`test_array_api`,
there is also a collection of Ivy tests, located in subfolder `test_ivy`_.

These tests serve two purposes:

#. test functions and classes which are *not* part of the standard
#. test additional required behaviour for functions which *are* part of the standard.
   The standard only mandates a subset of required behaviour, which the Ivy functions generally extend upon.

As done in the `test suite`_, we also make use of `hypothesis`_ for performing property based testing.

Hypothesis
----------

Using pytest fixtures (such as the ones removed in this `commit`_) cause a grid search to be performed for all
combinations of parameters. This is great when we want the test to be very thorough,
but can make the entire test suite very time consuming.
Before the changes in this commit, there were 300+ separate tests being run in total,
just for this :code:`ivy.abs` function.
If we take this approach for every function, we might hit the runtime limit permitted by GitHub actions.

A more elegant and efficient solution is to use the `hypothesis`_ module,
which intelligently samples from all of the possible combinations within user-specified ranges,
rather than grid searching all of them every single time.
The intelligent sampling is possible because hypothesis enables the results of previous test runs to be cached,
and then the new samples on subsequent runs are selected intelligently,
avoiding samples which previously passed the tests, and sampling for unexplored combinations.
Combinations which are known to have failed on previous runs are also repeatedly tested for.
With the `uploading`_ and `downloading`_ of the :code:`.hypothesis` cache as an `artifact`_,
these useful properties are also true in Ivy's GitHub Action `continuous integration`_ (CI) tests.

Rather than making use of :code:`pytest.mark.parametrize`, the Ivy tests make use of hypothesis `search strategies`_.
This reference `commit`_ outlines the difference between using pytest parametrizations and hypothesis,
for :code:`ivy.abs`.
Among other changes, all :code:`pytest.skip()` calls were replaced with return statements,
as pytest skipping does not play nicely with hypothesis testing.

Data Generation
---------------
We aim to make the data generation for three out of the four kinds of ivy functions exhaustive; primary, compositional
and mixed. Exhaustive data generation implies that all possible inputs and combinations of inputs are covered. Take
`finfo`_ , for example. It can take either arrays or dtypes as input, hence the `data generation`_ reflects this using
the bespoke search strategy :code:`_array_or_type`. However, such rigorous testing is not necessary for standalone functions
(those that are entirely self-contained in the Ivy codebase without external references). These kinds of functions may
only require standard Pytest testing using :code:`parametrize`, e.g. `test_default_int_dtype`_. For further clarity on
the various function types in ivy, see `here`_.

The way data is generated is described by the :code:`hypothesis.strategies` module which contains a variety of `methods`_
that have been used widely in each of Ivy's functional and stateful submodule tests. An initialized strategy is an object
that is used by Hypothesis to generate data for the test. For example, let's write a strategy that generates supported
integer data types in Ivy -:

.. code-block:: python

    valid_int_dtypes =  (int8,int16,int32,int64,uint8, uint16, uint32, uint64)
    custom_strategy = st.lists( st.sampled_from(valid_int_dtypes), min_size = 0, max_size = 4)

We are simply generating lists of arbitrary lengths within the range [0,4], wherein the elements correspond to the
valid_int_dtypes **tuple**. Let’s define a template function for printing examples generated by the hypothesis integrated
test functions.

**Note** - : This function will be referenced later in the section.

.. code-block:: python

    def print_hypothesis_examples(st: st.SearchStrategy, n = 2):
	    [print(f'Example run {i} -: {st.example()}') for i in range(0,n)]

Check the results of our strategy-:

.. code-block:: python

    print_hypothesis_examples( custom_strategy , 2)

    ['int8']
    ['int32', 'int16', 'int16', 'int32']

**Note** - : The output will be randomised in each run. This is quite a simplistic example and doesn’t cover the
intricacies behind the helper functions in the *test_ivy* directory.

In the example above, **st.lists** and **st.sampled_from** are what we call strategies. To briefly describe -:

1. `sampled_from`_ accepts a collection of objects. This strategy will return a value that is sampled from this
collection.

2. `lists`_ accepts another strategy which describes the elements of the list being generated. This is best used when
a sequence of varying lengths is required to be generated, with elements that are described by other strategies. The
following parameters can be specified by the user-:

* Bounds on the length of the list.
* If we want the elements to be unique.
* A mechanism for defining “uniqueness”.

Important Strategies
^^^^^^^^^^^^^^^^^^^^
It might be helpful to look at a few more strategies, since they are widely used across the  helper functions to
generate custom data -:

3. `booleans`_ - generates boolean values True or False

4. `integers`_ - generates integers values within a given range

5. `floats`_ -  It is a powerful strategy that generates all variety of floats, including math.inf and math.nan.
You can also specify:

* Math.inf and math.nan, respectively, should be included in the data description.
* Bounds(either inclusive or exclusive) on the floats being generated.
* The width of the floats; eg; if you want to generate 16-bit or 32 bit floats vs 64 bit. Python floats are always
  64-bit, width=32 ensures that the generated values can always be losslessly represented in both 32 bits. This is
  mostly useful for Numpy arrays).

6. `none`_ - returns a strategy which only generates None.

7. `tuples`_ - The strategy accepts N Hypothesis strategies, and will generate length - N tuples whose elements are drawn
from the respective strategies that were specified as inputs.

8. `one_of`_ - This allows us to specify a collection of strategies and any given datum will be drawn from “one of” them.
Hypothesis has the *pipe* operator overloaded as a shorthand for one_of. This has been widely used all over in Ivy Tests.
For example, this `line`_ here, can also be written as -:

.. code-block:: python

    st.one_of(st.none(), helpers.ints(min_value=-ndim, max_value=ndim -1))

9. `shared`_ - This returns a strategy that draws a shared value per run, drawn from base. Any two shared instances with
the same key will share the same value. For example, `here`_, the parameters, *input_dtype* and *as_variable* share
the same key *num_arrays*, hence similar values will be drawn for both arguments.

10. `sets`_ - This is used for generating a *unique collection* of elements. Like **st.lists** it accepts another strategy
which describes the elements of the set being generated.

11. `map`_ - The map method, permits us to perform a mapping on the data being produced by a strategy.

12. `filter`_ - Data is filtered using this method. It takes a callable that accepts as input the data generated by the
strategy, and returns:

* True if the data should pass through the filter
* False if the data should be rejected by the filter

13. `flatmap`_ - This enables us to define a strategy based on a value drawn from a previous strategy.

14. `data`_ - This is one of the **most** important strategies used in the project. It will often be the case that it is
required to draw strategies in a context-dependent manner within the test. Suppose, we want to generate an array of
values in some ivy test, but we want make sure that those values are only of the valid float types supported by Ivy.
The st.data() strategy can be used *interactively*, and values can be drawn at test-time, using **data.draw()** method.

The **given** operator usually contains the data parameter, which is an instance of the **st.DataObject** class; this
instance is what gets drawn from the st.data() strategy. For example, at `this`_ line the keyword arguments for the
function *test_argsort*, have been generated only after the generation of the array.

15. `composite`_ - The second **most** widely used strategy in *Ivy tests*. This provides a decorator, which permits us to
form our own strategies for describing data by composing Hypothesis’ built-in strategies. For `example`_.


Integration of Strategies into Ivy Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a strategy is initialised the **given** decorator is added to the test function for drawing values from the strategy
and passing them as inputs to the test. For example, in this code snippet here -:

.. code-block:: python

    @given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    native_array=st.booleans(),
    num_positional_args=helpers.ints(min_value=0, max_value=2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
    alpha=helpers.floats(),
    )
    def test_leaky_relu(
    dtype_and_x,
    alpha,
    as_variable,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
    ):
        dtype, x = dtype_and_x
        if not ivy.all(ivy.isfinite(ivy.array(x))) or not\
        ivy.isfinite(ivy.array([alpha])):
            return
        if fw == "torch" and dtype == "float16":
            return
        helpers.test_function(
   			dtype,
   			as_variable,
   			False,
   			native_array,
   			fw,
   			num_positional_args,
   			container,
   			instance_method,
   			"leaky_relu",
   			x=np.asarray(x, dtype=dtype),
   			alpha=alpha,)

In the test above, all parameters being exhaustively drawn inside the given block from hypothesis either
**directly** (*native_array, num_positional, instance_methods, alpha*) or **indirectly** (*dtype_and_x, as_variable, container*)
with the *helper* functions.

**Note** - It is advisable to specify the parameters of given as keyword arguments, so that there’s a correspondence
between our strategies with the function-signature’s parameters.

As  discussed above, the helper functions use the composite decorator, which helps in defining a series of custom strategies.
It can be seen that *dtype_and_x* uses the **dtype_and_values** strategy to generate valid float data types and corresponding
array elements, whose shapes can be specified manually or are assumed by default. The generated data is returned as a tuple.
Let's look at the data produced by this strategy -:

.. code-block:: python

    print_hypothesis_examples(dtype_and_values, 2)

    ('float64', [9433925.0, -1.401298464324817e-45])
    ('float64', [[574352379.0, -0.99999], [2.2250738585072014e-308, -6.103515625e-05]])

These values are then unpacked, converted to :code:`ivy.array` class, with corresponding dtypes. The test then runs on the newly
created arrays with specified dtypes. Similar is the case with other parameters which the function above is required to test.

Why do we need helper functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is usually the case that any ivy function should run seamlessly on ‘all the possible varieties, as well as  the edge
cases’ encountered by the following parameters -:

* All possible data types - **composite**
* Boolean array types if the function expects one - **composite**
* Possible range of values within each data type - **composite**
* When input is a container - **boolean**
* When the function can also be called as an instance method - **boolean**
* When the input is a native array - **boolean**
* Out argument support, if the function has one - **boolean**

**Note** -: Each test function has its own requirements and the parameter criterion listed above does not cover everything.

Sometimes the function requirements are straight-forward, for instance, generating integers, boolean values, float values.
Whereas, in the case of specific parameters like -:

* array_values
* data_types
* valid_axes
* lists or tuples or sequence of varied input types( the test_leaky_relu function above)
* generating subsets at test time
* generating arbitrary shapes of arrays at test time
* getting axes at test time

We need a hand-crafted data generation policy(composite). For this purpose ad-hoc functions have been defined in the
:code:`helpers.py` file. It might be appropriate now, to bring them up and discuss their use. A detailed overview of their working
is as follows-:

1. **array_dtypes** - As the name suggests, this will generate arbitrary sequences of valid float data types. The sequence
parameters like *min_size*, and *max_size*, are specified at test time based on the function. This is what the function
returns -:

.. code-block:: python

    #a sequence of floats with arbitrary lengths ranging from [1,5]
    print_hypothesis_examples(array_dtypes(helpers.ints(min_value=1, max_value=5)))

    ['float16', 'float32', 'float16', 'float16', 'float32']
    ['float64', 'float64', 'float32', 'float32', 'float16']

This function should be used whenever we are testing an ivy function that accepts at least one array as an input.

2. **array_bools** - This function generates a sequence of boolean values. For example-:

.. code-block:: python

    print_hypothesis_examples(array_bools(na = helpers.ints(min_value=1, max_value=5)))

    [False, True, True, False, True]
    [False]

This function should be used when a boolean value is to be associated for each value of the other parameter, when
generated by a sequence. For example, in `test_concat`_, we are generating a list of inputs of the dimension (2,3), and
for each input we have three boolean values associated with it that define additional parameters(container, as_variable
, native_array). Meaning if the input is to be treated as a container, at the same time, is it a variable or a native array.

3. **lists** - As the name suggests, we use it to generate lists composed of anything, as specified by the user. For example
in `test_device`_ file, it is used to generate a list of array_shapes, in `test_manipulation`_, it is used to generate a list
of common_shapes, and more in `test_layers`_. The function takes in 3 arguments, first is the strategy by which the elements
are to be generated, in majority of the cases this is **helpers.ints**, with range specified, and the other arguments are
sequence arguments as specified in **array_dtypes**. For example -:

.. code-block:: python

    print_hypothesis_examples(lists(helpers.ints(min_value=1, max_value=6), min_size = 0,max_size = 5))

    [2, 5, 6]
    [1]

The generated values are then passed to the array creation functions inside the test function as tuples.

4. **valid_axes** - This function generates valid axes for a given array dimension. For example -:

.. code-block:: python

    print_hypothesis_examples(valid_axes(helpers.ints(min_value=2, max_value=3), size_bounds = [1,3]))

    (-3, 1, -1)
    (1, -2)

It should be used in functions which expect axes as a required or an optional argument.

5. **integers** - This is similar to the *helpers.ints* strategy, with the only difference being that here the range can
either be specified manually, or a shared key can be provided. The way shared keys work has been discussed in the
*Important Strategies* sections above.

6. **dtype_and_values** - This function generates a tuple wherein the first element is a valid float data type, and the
second element is a list/nested list containing floating point numbers of that precision. For example-:

.. code-block:: python

    #ivy valid float types are those which are supported by numpy
    import ivy.functional.backends.numpy as ivy_np
    print_hypothesis_examples(dtype_and_values(ivy_np.valid_float_dtypes), 3)

    ('float64', 0.0)
    ('float16', 0.0)
    ('float64', [283405296074752.0, 564049465049088.0, 1.0417876997507982e+16])

This function contains a list of `keyword`_ arguments. To name a few, min_value, max_value, allow_inf, min_num_dims etc.
It can be used wherever an array of values with a specified data type is expected. That would again be a list a functions
which expects at least one :code:`ivy.array`.

7. **reshape_shapes** - This function returns a valid shape after a reshape operation is applied given as input of any
arbitrary shape. For example-:

.. code-block:: python

   print_hypothesis_examples(reshape_shapes([3,3]), 3)

   (9, 1)
   (9,)
   (-1,)

It should be used in places where broadcast operations are run, either as a part of a larger computation or in a
stand-alone fashion.

8. **subsets** - As the function name suggests, it generates subsets of any sequence, and returns that subset as a tuple.
For example-:

.. code-block:: python

    some_sequence = ['tensorflow', 1, 3.06, 'torch', 'ivy', 0]
    print_hypothesis_examples(subsets(some_sequence), 4)

    ('tensorflow', 'ivy', 0)
    ('tensorflow', 1, 3.06, 'torch', 'ivy')
    ('tensorflow', 1, 'torch', 0)
    (1, 3.06)

9. **array_values** - It works in a similar way as the **dtype_and_values** function, with the only difference being,
here an extensive set of parameters and sub-strategies are used to generate array values. For example-:

.. code-block:: python

    input_dtype = st.sampled_from(ivy_np.valid_float_dtypes)
    print_hypothesis_examples(
                              array_values(
                              input_dtype.example(), shape=(3,),
 	                          min_value=0,   allow_subnormal = True,
                              exclude_min=True
                                          )
                              )

    [5.960464477539063e-08, 5.960464477539063e-08, 0.5]
    [5.960464477539063e-08, 5.960464477539063e-08, 1.0]

It ensures full coverage of the values that an array can have, given certain parameters like *allow_nan, allow_subnormal, allow_inf*.
Such parameters usually test the function for edge cases. This function should be used in places where the result doesn’t
depend on the kind of value an array contains.

10. **get_shape** - This is used to generate any arbitrary shape. If *allow_none* is set to :code:`True`, then an implicit
*st.one_of* strategy is used, wherein the function will either generate :code:`None` as shape or it will generate a shape
based on the keyword `arguments`_ of the function. For example -:

.. code-block:: python

    print_hypothesis_examples(
                              get_shape(
                              allow_none = True, min_num_dims = 2,
                              max_num_dims = 7, min_dim_size = 2
                                       ), 3
                              )
    (5, 5, 8)
    (4, 3, 3, 4, 9, 9, 8)
    (9, 9, 3, 5, 6)

11. **none_or_list_of_floats** - This function is the same as array_values function, with the only difference being that here
data types other than float are not supported. User needs to pass in a *valid float type*, and the *size*. Here :code:`None`
type is :code:`True` by default. For example-:

.. code-block:: python

    print_hypothesis_examples(
                              none_or_list_of_floats(
                              input_dtype.example(), size = 5,
                              min_value=10.0, max_value= 200.0),3
                              )
    [None, 199.99999999999997, 200.0, None, 199.99999999999997]
    [199.99999999999997, None, None, 10.000000000000002, 125.43759670925832]
    [None, 10.0, 199.0, 10.0, 200.0]

This function might come in handy when some float values are required for generating other data, or are part of a larger
computation. For example, **get_mean_std** strategy requires a series of values to generate the mean and standard deviation
for arbitrary input values.

12. **get_mean_std** - Strategies like this one are specific to a particular range of functions only. It comes in handy while
testing probabilistic functions like *random_normal*, and other distributions or statistical functions like *mean-squared-error*.
For example-:

.. code-block:: python

    input_dtype = st.sampled_from(ivy_np.valid_float_dtypes)
    print_hypothesis_examples(get_mean_std(input_dtype.example()))

    (0.0, None)
    (9.811428143185347e+89, None)

**Note** - This strategy uses **none_or_list_floats** internally, and so the standard deviation and mean may or may not
be None.

13. **get_bounds** -  It’s often the case that we need to define a lower and an upper limit for generating certain values,
like floats, sequences, arrays_values etc. This strategy can be put to use when we want our function to pass on values
in any range  possible, or we’re unsure about the limits. We can also use the function to generate a list of possible
bounds wherein the function fails. For example-:

.. code-block:: python

    input_dtype = st.sampled_from(ivy_np.valid_int_dtypes)
    print_hypothesis_examples(get_bounds(input_dtype.example()))

    (73, 36418)
    (213, 21716926)

**Note** - Under the hood, **array_values** strategy is called if the data type is *integer*, and **none_or_list_of_floats**
is called when the data type is *float*.

14. **get_probs** -  This is similar to the **get_mean_std** strategy, and is used to generate a tuple containing two values.
The first one being the *unnormalized probabilities* for all elements in a population, the second one being the *population size*.
For example-:

.. code-block:: python

   input_dtype = st.sampled_from(ivy_np.valid_float_dtypes)
   print_hypothesis_examples(get_probs(input_dtype.example()))

   ([[6.103515625e-05, 1.099609375], [1.0, 6.103515625e-05], [1.0, 1.0], [0.5, 6.103515625e-05]], 2)

Such strategies can be used to test statistical and probabilistic functions in Ivy.

15. **get_axis** - Similar to the **valid_axes** strategy, it generates an axis given any arbitrary shape as input.
For example-:

.. code-block:: python

    print_hypothesis_examples(get_axis(shape = (3,3,2)))

    (-1,)
    (-2, -1)

16. **num_positional_args** - A helper function which generates the number of positional arguments, provided a function name
from any ivy submodule. For example -:

.. code-block:: python

    print_hypothesis_examples(num_positional_args("matmul"), 3)

    2
    0
    0

This function generates any number of positional arguments within the range [0, number_positional_arguments]. It can be
helpful when we are testing a function with varied number of arguments.


How to write Hypothesis Tests effectively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It would be helpful to keep in mind the following points while writing test -:

a. Don't use :code:`data.draw` in the function body.
b. Don't use array generation (i.e. np.random_uniform) in the function body.
c. Don't skip anything in the function body.
d. The function should only call helpers.test_function, and then possibly perform a custom value test if
   :code:`test_values=False` in the arguments.
e. We should add as many possibilities as we can while generating data, covering all the function arguments
f. If you find yourself using repeating some logic which is specific to a particular submodule, then create a private
   helper function and add this to the submodule.
g. If the logic is general enough, this can instead be added to the :code:`helpers.py` file, enabling it to be used for tests
   in other submodules
h. Sometimes, the use of
   `assume <https://hypothesis.readthedocs.io/en/latest/details.html?highlight=assume#hypothesis.assume>`_
   is justified in the unit test body, particularly for cases where writing the
   generation code would be unduly laborious. It's very straightforward to avoid
   :code:`nan`, :code:`inf` and values close to the :code:`dtype` bounds, but also
   avoiding zeros would require extra implementational effort in the data generation
   helpers. Using :code:`assume` is an
   `acceptable solution <https://github.com/unifyai/ivy/blob/2ddaff94ad9e20a1a0511d272a0501fa3b904edc/ivy_tests/test_ivy/test_functional/test_core/test_elementwise.py#L695>`_
   in such cases, and other similar scenarios you may encounter.


Bonus: Hypothesis' Extended Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Hypothesis** performs **Automated Test-Case Reduction**. That is, the **given** decorator strives to report the simplest
set of input values that produce a given error. For the code block below-:

.. code-block:: python

    @given(
    data = st.data(),
    input_dtype = st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans()
    )
    def test_demo(
       data,
       input_dtype,
       as_variable,
    ):
        shape = data.draw(get_shape(min_num_dims=1))

        #failing assertions
        assert as_variable == False
        assert shape == 0

    test_demo()

Hypothesis reports the following -:

.. code-block:: python

    Falsifying example: failing_test(
    data=data(...), input_dtype='float16', as_variable=True,
    )
    Draw 1: (1,)
    Traceback (most recent call last):
    File "<file_name>.py" line "123", in test_demo
    assert as_variable == False
    AssertionError

    Falsifying example: failing_test(
    data=data(...), input_dtype='float16', as_variable=False,
    )
    Draw 1: (1,)
    assert shape == 0
    AssertionError

As can be seen from the output above, the given decorator will report the *simplest* set of input values that produce a
given error. This is done through the process of **Shrinking**.

Each of the Hypothesis’ strategies has it’s own prescribed shrinking behavior. For integers, it will identify the integer
closest to 0 that produces the error at hand. Checkout the `documentation`_ for more information on shrinking behaviors of
other strategies.

Hypothesis doesn’t search for falsifying examples from scratch every time the test is run. Instead, it save a database of
these examples associated with each of the project’s test functions. In the case of Ivy, the :code:`.hypothesis` cache
folder is generated if one doesn’t exist, otherwise the existing one is added to it. We just preserve this folder on the
CI, so that each commit uses the same folder, and so it is ignored by git, thereby never forming part of the :code:`commit`.

2. **–-hypothesis-show-statistics**

This feature helps is debugging the tests, with methods like **note()**, custom **event()s** where addition to the summary,
and a variety performance details are supported. Let’s look at the function `test_gelu`_ -:

**run** :code:`pytest —hypothesis-show-statistics <test_file>.py`

This test runs for every backend, and the output is shown below-:

* **Jax**
.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/Jax_data_gen.png
   :width: 600

* **Numpy**
.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/numpy_data_gen.png
   :width: 600

* **Tensorflow**
.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/tensorflow_data_gen.png
   :width: 600

* **Torch**
.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/torch_data_gen.png
   :width: 600


It can be seen that the function doesn’t fail for **Jax**, **Numpy** and **Torch**, which is clearly not the case with
**Tensorflow**, wherein 7 examples failed the test. One important thing to note is the number of values for which
**Shrinking**(discussed in brief above) happened. Statistics for both *generate phase*, and *shrink phase* if the test
fails are printed in the output. If the tests are re-run, *reuse phase* statistics are printed as well where notable
examples from previous runs are displayed.

Another argument which can be specified for a more detailed output is **hypothesis-verbosity = verbose**. Let’s look at
the newer output, for the same example -:

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/test_run_data_gen.png
   :width: 600

Like the output above, Hypothesis will print all the examples for which the test failed, when **verbosity** is set.


3. Some performance related settings which might be helpful to know are-:

a. **max_examples** - The number of valid examples Hypothesis will run. It usually defaults to 100. Turning it up or down
                      will have an impact on the speed as well as the rigorousness of the tests.

b. **deadline** - If an input takes longer than expected, it should be treated as an error. It is useful to detect weird
                  performance issues.

Self-Consistent and Explicit Testing
------------------------------------

The hypothesis data generation strategies ensure that we test for arbitrary variations in the function inputs,
but this makes it difficult to manually verify ground truth results for each input variation.
Therefore, we instead opt to test for self-consistency against the same Ivy function with a NumPy backend.
This is handled by :code:`test_array_function`, which is a helper function most unit tests defer to.
This function is explained in more detail in the following sub-section.

For *primary* functions, this approach works well.
Each backend implementation generally wraps an existing backend function,
and under the hood these implementations vary substantially.
This approach then generally suffices to correctly catch bugs for most *primary* functions.

However, for *compositional* and *mixed* functions, then it's more likely that a bug could be missed.
With such functions, it's possible that the bug exists in the shared *compositional* implementation,
and then the bug would be systematic across all backends,
including the *ground truth* NumPy which the value tests for all backends compare against.

Therefore, for all *mixed* and *compositional* functions,
the test should also be appended with known inputs and known ground truth outputs,
to safeguard against this inability for :code:`test_array_function` to catch systematic errors.
These should be added using :code:`pytest.mark.parametrize`.
However, we should still also include :code:`test_array_function` in the test,
so that we can still test for arbitrary variations in the input arguments.

test_array_function
-------------------

The helper `test_array_function`_ tests that the function:

#. can handle the :code:`out` argument correctly
#. can be called as an instance method of the ivy.Array class
#. can accept ivy.Container instances in place of any arguments for *nestable* functions,
   applying the function to the leaves of the container, and returning the resultant container
#. can be called as an instance method on the ivy.Container
#. is self-consistent with the function return values when using a NumPy backend

:code:`array` in the name :code:`test_array_function` simply refers to the fact that the function in question consumes
arrays in the arguments.

So when should :code:`test_array_function` be used?

The rule is simple, if the test should not pass any arrays in the input,
then we should not use the helper :code:`test_array_function`.
For example, :code:`ivy.num_gpus` does not receive any arrays in the input,
and so we should not make us of :code:`test_array_function` in the test implementation.

**Round Up**

This should have hopefully given you a good feel for how the tests are implemented in Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy tests discussion`_,
or reach out on `discord`_ in the `ivy tests channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/E6WgGp2_e5E" class="video">
    </iframe>