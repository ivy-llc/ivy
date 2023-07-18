Ivy Frontend Tests
==================

.. _`here`: https://unify.ai/docs/ivy/design/ivy_as_a_transpiler.html
.. _`ivy frontends tests channel`: https://discord.com/channels/799879767196958751/1028267758028337193
.. _`ivy frontends tests forum`: https://discord.com/channels/799879767196958751/1028297887605587998
.. _`test ivy`: https://github.com/unifyai/ivy/tree/db9a22d96efd3820fb289e9997eb41dda6570868/ivy_tests/test_ivy
.. _`test_frontend_function`: https://github.com/unifyai/ivy/blob/591ac37a664ebdf2ca50a5b0751a3a54ee9d5934/ivy_tests/test_ivy/helpers.py#L1047
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`Function Wrapping`: https://unify.ai/docs/ivy/overview/deep_dive/function_wrapping.html
.. _`open task`: https://unify.ai/docs/ivy/overview/contributing/open_tasks.html
.. _`Ivy Tests`: https://unify.ai/docs/ivy/overview/deep_dive/ivy_tests.html
.. _`Function Testing Helpers`: https://github.com/unifyai/ivy/blob/bf0becd459004ae6cffeb3c38c02c94eab5b7721/ivy_tests/test_ivy/helpers/function_testing.py
.. _`CI Pipeline`: https://unify.ai/docs/ivy/overview/deep_dive/continuous_integration.html
.. _`setting up`: https://unify.ai/docs/ivy/compiler/setting_up.html#setting-up-testing


Introduction
------------

Just like the backend functional API, our frontend functional API has a collection of Ivy tests located in subfolder `test ivy`_.
In this section of the deep dive we are going to jump into Ivy Frontend Tests!

**Writing Ivy Frontend Tests**

The Ivy tests in this section make use of hypothesis for performing property based testing which is documented in detail in the Ivy Tests section of the Deep Dive.
We assume knowledge of hypothesis data generation strategies and how to implement them for testing.

**Ivy Decorators**

Ivy provides test decorators for frontend tests to make it easier and more maintainable, currently there are two:

* :func:`@handle_frontend_test` a decorator which is used to test frontend functions, for example :func:`np.zeros` and :func:`tensorflow.tan`.
* :func:`@handle_frontend_method` a decorator which is used to test frontend methods and special methods, for example :func:`torch.Tensor.add` and :func:`numpy.ndarray.__add__`.

**Important Helper Functions**

* :func:`helpers.test_frontend_function` helper function that is designed to do the heavy lifting and make testing Ivy Frontends easy!
  One of the many `Function Testing Helpers`_.
  It is used to test a frontend function for the current backend by comparing the result with the function in the associated framework.

* :func:`helpers.get_dtypes` helper function that returns either a full list of data types or a single data type, we should **always** be using `helpers.get_dtypes` to sample data types.

* :func:`helpers.dtype_and_values` is a convenience function that allows you to generate arrays of any dimension and their associated data types, returned as :code:`([dtypes], [np.array])`.

* :func:`helpers.get_shape` is a convenience function that allows you to generate an array shape of type :code:`tuple`

* :func:`np_frontend_helpers.where` a generation strategy to generate values for NumPy's optional :code:`where` argument.

* :func:`np_frontend_helpers.test_frontend_function` behaves identical to :func:`helpers.test_frontend_function` but handles NumPy's optional :code:`where` argument

**Useful Notes**

* We should always ensure that our data type generation is complete.
  Generating float data types only for a function that accepts all numeric data types is not complete, a complete set would include **all** numeric data types.

* The :func:`test_frontend_function` argument :code:`fn_tree` refers to the frontend function's reference in its native namespace not just the function name.
  For example :func:`lax.tan` is needed for some functions in Jax, :func:`nn.functional.relu` is needed for some functions in PyTorch etc.

To get a better understanding for writing frontend tests lets run through some examples!

Frontend Test Examples
-----------------------

Before you begin writing a frontend test, make sure you are placing it in the correct location.
See the 'Where to place a frontend function' sub-section of the frontend APIs `open task`_ for more details.

ivy.tan()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @handle_frontend_test(
        fn_tree="jax.lax.tan",
        dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
        test_with_out=st.just(False),
    )
    def test_jax_lax_tan(
        *,
        dtype_and_x,
        on_device,
        fn_tree,
        frontend,
        test_flags,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            x=x[0],
        )

* As you can see we generate almost everything we need to test a frontend function within the :code:`@handle_frontend_test` decorator.
* We set :code:`fn_tree` to :code:`jax.lax.tan` which is the path to the function in the Jax namespace.
* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid :code:`float` data types specifically for Jax.
* We do not generate any values for :code:`as_variable`, :code:`native_array`, :code:`frontend`, :code:`num_positional_args`, :code:`on_device`, these values are generated by :func:`handle_frontend_test`.
* We unpack the :code:`dtype_and_x` to :code:`input_dtype` and :code:`x`.
* We then pass the generated values to :code:`helpers.test_frontend_function` which tests the frontend function.
* :func:`jax.lax.tan` does not support :code:`out` arguments so we set :code:`with_out` to :code:`False`.
* One last important note is that all helper functions are designed to take keyword arguments only.

**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_mathematical_functions/test_np_trigonometric_functions.py
        @handle_frontend_test(
        fn_tree="numpy.tan",
        dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("float"),
                )
            ],
        ),
        where=np_frontend_helpers.where(),
        number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
                fn_name="tan"
            ),
        )
        def test_numpy_tan(
            dtypes_values_casting,
            where,
            frontend,
            test_flags,
            fn_tree,
            on_device,
        ):
            input_dtypes, x, casting, dtype = dtypes_values_casting
            where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
                where=where,
                input_dtype=input_dtypes,
                test_flags=test_flags,
            )
            np_frontend_helpers.test_frontend_function(
                input_dtypes=input_dtypes,
                frontend=frontend,
                test_flags=test_flags,
                fn_tree=fn_tree,
                on_device=on_device,
                x=x[0],
                out=None,
                where=where,
                casting=casting,
                order="K",
                dtype=dtype,
                subok=True,
            )
* We set :code:`fn_tree` to :code:`numpy.tan` which is the path to the function in the NumPy namespace.
* Here we use :code:`helpers.get_dtypes("numeric")` to generate :code:`available_dtypes`, these are valid :code:`numeric` data types specifically for NumPy.
* NumPy has an optional argument :code:`where` which is generated using :func:`np_frontend_helpers.where`.
* Using :func:`np_frontend_helpers.handle_where_and_array_bools` we do some processing on the generated :code:`where` value.
* Instead of :func:`helpers.test_frontend_function` we use :func:`np_frontend_helpers.test_frontend_function` which behaves the same but has some extra code to handle the :code:`where` argument.
* :code:`casting`, :code:`order`, :code:`subok` and other are optional arguments for :func:`numpy.tan`.

**TensorFlow**

.. code-block:: python
        
        # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_math.py
        # tan
        @handle_frontend_test(
            fn_tree="tensorflow.math.tan",
            dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
            test_with_out=st.just(False),
        )
        def test_tensorflow_tan(
            *,
            dtype_and_x,
            frontend,
            test_flags,
            fn_tree,
            on_device,
        ):
            input_dtype, x = dtype_and_x
            helpers.test_frontend_function(
                input_dtypes=input_dtype,
                frontend=frontend,
                test_flags=test_flags,
                fn_tree=fn_tree,
                on_device=on_device,
                x=x[0],
            )
* We set :code:`fn_tree` to :code:`tensorflow.math.tan` which is the path to the function in the TensorFlow namespace.
* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for the function.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_pointwise_ops.py
    # tan
    @handle_frontend_test(
    fn_tree="torch.tan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
            ),
    )
    def test_torch_tan(
        *,
        dtype_and_x,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        on_device,
        fn_tree,
        frontend,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend=frontend,
            fn_tree=fn_tree,
            on_device=on_device,
            input=x[0],
        )

* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for the function.

ivy.full()
^^^^^^^^^^

Here we are going to look at an example of a function that does not consume an :code:`array`.
This is the creation function :func:`full`, which takes an array shape as an argument to create an array filled with elements of a given value.
This function requires us to create extra functions for generating :code:`shape` and :code:`fill value`, these use the :code:`shared` hypothesis strategy.


**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        elif ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))


    @handle_frontend_test(
        fn_tree="jax.lax.full",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
    )
    def test_jax_lax_full(
        *,
        shape,
        fill_value,
        dtypes,
        on_device,
        fn_tree,
        frontend,
        test_flags,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            shape=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
        )


* The custom function we use is :code:`_fill_value` which generates a :code:`fill_value` to use for the :code:`fill_value` argument but handles the complications of :code:`int` and :code:`uint` types correctly.
* We use the helper function :func:`helpers.get_shape` to generate :code:`shape`.
* We use :code:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Jax.
  This is used to specify the data type of the output array.
* :func:`full` does not consume :code:`array`.


**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/creation_routines/test_from_shape_or_value.py
    @st.composite
    def _input_fill_and_dtype(draw):
        dtype = draw(helpers.get_dtypes("float", full=False))
        dtype_and_input = draw(helpers.dtype_and_values(dtype=dtype))
        if ivy.is_uint_dtype(dtype[0]):
            fill_values = draw(st.integers(min_value=0, max_value=5))
        elif ivy.is_int_dtype(dtype[0]):
            fill_values = draw(st.integers(min_value=-5, max_value=5))
        else:
            fill_values = draw(st.floats(min_value=-5, max_value=5))
        dtype_to_cast = draw(helpers.get_dtypes("float", full=False))
        return dtype, dtype_and_input[1], fill_values, dtype_to_cast[0]

    # full
    @handle_frontend_test(
        fn_tree="numpy.full",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        input_fill_dtype=_input_fill_and_dtype(),
        test_with_out=st.just(False),
    )
    def test_numpy_full(
        shape,
        input_fill_dtype,
        frontend,
        test_flags,
        fn_tree,
        on_device,
    ):
        input_dtype, x, fill, dtype_to_cast = input_fill_dtype
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            shape=shape,
            fill_value=fill,
            dtype=dtype_to_cast,
        )

* We use :func:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for NumPy.
* :func:`numpy.full` does not have a :code:`where` argument so we can use :func:`helpers.test_frontend_function`, we specify the `out` flag explicitely.

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tf_functions.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    # fill
    @handle_frontend_test(
        fn_tree="tensorflow.raw_ops.Fill",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            min_dim_size=1,
        ),
        fill_value=_fill_value(),
        dtypes=_dtypes(),
        test_with_out=st.just(False),
    )
    def test_tensorflow_Fill(  # NOQA
        *,
        shape,
        fill_value,
        dtypes,
        frontend,
        test_flags,
        fn_tree,
        on_device,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            rtol=1e-05,
            dims=shape,
            value=fill_value,
        )


* We use :func:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for this function.
* Tensorflow's version of :func:`full` is named :func:`Fill` therefore we specify the :code:`fn_tree` argument to be :code:`"Fill"`
* When running the test there were some small discrepancies between the values so we can use :code:`rtol` to specify the relative tolerance. We specify the `out` flag explicitely.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_creation_ops.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_frontend_test(
        fn_tree="torch.full",
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtype=st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"),
    )
    def test_torch_full(
        *,
        shape,
        fill_value,
        dtype,
        on_device,
        fn_tree,
        frontend,
        test_flags,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtype,
            on_device=on_device,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            size=shape,
            fill_value=fill_value,
            dtype=dtype[0],
            device=on_device,
        )

* We use :code:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Torch.

Testing Without Using Tests Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While even using hypothesis, there are some cases in which we set :code:`test_values=False` for example, we have a
function add_noise() and we call it on x and we try to assert (we interally use assert np.all_close) that the result
from torch backend matches tensorflow and the test will always fail, because the function add_noise() depends on a random
seed internally that we have no control over, what we change is only how we test for equality, in which in that case
we can not and we have to reconstruct the output as shown in the example below.

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_linalg.py
    @handle_frontend_test(
        fn_tree="torch.linalg.qr",
        dtype_and_input=_get_dtype_and_matrix(),
        test_with_out=st.just(False),
    )
    def test_torch_qr(
        *,
        dtype_and_input,
        frontend,
        test_flags,
        fn_tree,
        on_device,
    ):
        input_dtype, x = dtype_and_input
        ret, frontend_ret = helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            input=x[0],
            test_values=False,
        )
        ret = [ivy.to_numpy(x) for x in ret]
        frontend_ret = [np.asarray(x) for x in frontend_ret]

        q, r = ret
        frontend_q, frontend_r = frontend_ret

        assert_all_close(
            ret_np=q @ r,
            ret_from_gt_np=frontend_q @ frontend_r,
            rtol=1e-2,
            atol=1e-2,
            ground_truth_backend=frontend,
        )

* The parameter :code:`test_values=False` is explicitly set to "False" as there can be multiple solutions for this and those multiple solutions can all be correct, so we have to test with reconstructing the output.

What assert_all_close() actually does is, it checks for values and dtypes, if even one of them is not same it will cause
an assertion, the examples given below will make it clearer.

.. code-block:: python

    >>> a = np.array([[1., 5.]], dtype='float32')
    >>> b = np.array([[2., 4.]], dtype='float32')
    >>> print(helpers.assert_all_close(a, b))
    AssertionError: [[1. 5.]] != [[2. 4.]]


.. code-block:: python

    >>> a = np.array([[1., 5.]], dtype='float64')
    >>> b = np.array([[2., 4.]], dtype='float32')
    >>> print(helpers.assert_all_close(a, b))
    AssertionError: the return with a TensorFlow backend produced data type of float32, while the return with a  backend returned a data type of float64.


Alias functions
^^^^^^^^^^^^^^^
Let's take a quick walkthrough on testing the function alias as we know that such functions have the same behavior as original functions.
Taking an example of :func:`torch_frontend.greater` has an alias function :func:`torch_frontend.gt` which we need to make sure that it is working same as the targeted framework function :func:`torch.greater` and :func:`torch.gt`.

Code example for alias function:

.. code-block:: python

    # in ivy/functional/frontends/torch/comparison_ops.py
    @to_ivy_arrays_and_back
    def greater(input, other, *, out=None):
        input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
        return ivy.greater(input, other, out=out


    gt = greater

* As you can see the :func:`torch_frontend.gt` is an alias to :func:`torch_frontend.greater` and below is how we update the unit test of :func:`torch_frontend.greater` to test the alias function as well.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_comparison_ops.py
    @handle_frontend_test(
        fn_tree="torch.gt",
        aliases=["torch.greater"],
        dtype_and_inputs=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=2,
            allow_inf=False,
            shared_dtype=True,
        ),
    )
    def test_torch_greater(
        *,
        dtype_and_inputs,
        on_device,
        fn_tree,
        frontend,
        test_flags,
    ):
        input_dtype, inputs = dtype_and_inputs
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            input=inputs[0],
            other=inputs[1],
        )

* We added a list of all aliases to the :code:`greater` function with a full namespace path such that when we are testing the original function we will test for the alias as well.
* During the frontend implementation, if a new alias is introduced you only need to go to the test function of the original frontend function and add that alias to :code:`all_aliases` argument in the :func:`test_frontend_function` helper with its full namespace.

Frontend Instance Method Tests
------------------------------

The frontend instance method tests are similar to the frontend function test, but instead of testing the function directly we test the instance method of the frontend class.
major difference is that we have more flags to pass now, most initialization functions take an array as an input. also some methods may take an array as input,
for example, :code:`ndarray.__add__` would expect an array as input, despite the :code:`self.array`. and to make our test **complete** we need to generate seperate flags for each.

**Important Helper Functions**

:func:`@handle_frontend_method` requires 3 keyword only parameters:
    - :code:`class_tree` A full path to the array class in **Ivy** namespace. 
    - :code:`init_tree` A full path to initialization function.
    - :code:`method_name` The name of the method to test. 

:func:`helpers.test_frontend_method` is used to test frontend instance methods. It is used in the same way as :func:`helpers.test_frontend_function`.


Frontend Instance Method Test Examples
--------------------------------------

ivy.add()
^^^^^^^^^
**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_ndarray.py
    @handle_frontend_method(
        class_tree=CLASS_TREE,
        init_tree="numpy.array",
        method_name="__add__",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
        ),
    )
    def test_numpy_instance_add__(
        dtype_and_x,
        frontend_method_data,
        init_flags,
        method_flags,
        frontend,
    ):
        input_dtypes, xs = dtype_and_x

        helpers.test_frontend_method(
            init_input_dtypes=input_dtypes,
            init_all_as_kwargs_np={
                "object": xs[0],
            },
            method_input_dtypes=input_dtypes,
            method_all_as_kwargs_np={
                "value": xs[1],
            },
            frontend=frontend,
            frontend_method_data=frontend_method_data,
            init_flags=init_flags,
            method_flags=method_flags,
        )


* We specify the :code:`class_tree` to be :meth:`ivy.functional.frontends.numpy.array` which is the path to the class in ivy namespace.
* We specify the function that is used to initialize the array, for jax, we use :code:`numpy.array` to create a :code:`numpy.ndarray`.
* We specify the :code:`method_name` to be :meth:`__add__` which is the path to the method in the frontend class.

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tensor.py
    @handle_frontend_method(
        class_tree=CLASS_TREE,
        init_tree="tensorflow.constant",
        method_name="__add__",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=2,
            shared_dtype=True,
        ),
    )
    def test_tensorflow_instance_add(
        dtype_and_x,
        frontend,
        frontend_method_data,
        init_flags,
        method_flags,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_method(
            init_input_dtypes=input_dtype,
            init_all_as_kwargs_np={
                "value": x[0],
            },
            method_input_dtypes=input_dtype,
            method_all_as_kwargs_np={
                "y": x[1],
            },
            frontend=frontend,
            frontend_method_data=frontend_method_data,
            init_flags=init_flags,
            method_flags=method_flags,
        )


* We specify the function that is used to initialize the array, for TensorFlow, we use :code:`tensorflow.constant` to create a :code:`tensorflow.EagerTensor`.
* We specify the :code:`method_tree` to be :meth:`tensorflow.EagerTensor.__add__` which is the path to the method in the frontend class.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_tensor.py
    @handle_frontend_method(
        class_tree=CLASS_TREE,
        init_tree="torch.tensor",
        method_name="add",
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=2,
            min_value=-1e04,
            max_value=1e04,
            allow_inf=False,
        ),
        alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
    )
    def test_torch_instance_add(
        dtype_and_x,
        alpha,
        frontend,
        frontend_method_data,
        init_flags,
        method_flags,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_method(
            init_input_dtypes=input_dtype,
            init_all_as_kwargs_np={
                "data": x[0],
            },
            method_input_dtypes=input_dtype,
            method_all_as_kwargs_np={
                "other": x[1],
                "alpha": alpha,
            },
            frontend_method_data=frontend_method_data,
            init_flags=init_flags,
            method_flags=method_flags,
            frontend=frontend,
            atol_=1e-02,
        )


* We specify the function that is used to initialize the array, for PyTorch, we use :code:`torch.tensor` to create a :code:`torch.Tensor`.
* We specify the :code:`method_tree` to be :meth:`torch.Tensor.__add__` which is the path to the method in the frontend class.

Hypothesis Helpers
------------------

Naturally, many of the functions in the various frontend APIs are very similar to many of the functions in the Ivy API.
Therefore, the unit tests will follow very similar structures with regards to the data generated for testing.
There are many data generation helper functions defined in the Ivy API test files, such as :func:`_arrays_idx_n_dtypes` defined in :mod:`ivy/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py`.
This helper generates: a set of concatenation-compatible arrays, the index for the concatenation, and the data types of each array.
Not surprisingly, this helper is used for testing :func:`ivy.concat`, as shown `here <https://github.com/unifyai/ivy/blob/86287f4e45bbe581fe54e37d5081c684130cba2b/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py#L53>`_.

Clearly, this helper would also be very useful for testing the various frontend concatenation functions, such as :code:`jax.numpy.concatenate`, :code:`numpy.concatenate`, :code:`tensorflow.concat` and :code:`torch.cat`.
We could simply copy and paste the implementation from :mod:`/ivy_tests/test_ivy/test_functional/test_core/test_manipulation.py` into each file :mod:`/ivy_tests/test_ivy/test_frontends/test_<framework>/test_<group>.py`, but this would result in needless duplication.
Instead, we should simply import the helper function from the ivy test file into the frontend test file, like so :code:`from ivy_tests.test_ivy.test_frontends.test_manipulation import _arrays_idx_n_dtypes`.

In cases where a helper function is uniquely useful for a frontend function without being useful for an Ivy function, then it should be implemented directly in :mod:`/ivy_tests/test_ivy/test_frontends/test_<framework>/test_<group>.py` rather than in :mod:`/ivy_tests/test_ivy/test_functional/test_core/test_<closest_relevant_group>.py`.
However, as shown above, in many cases the same helper function can be shared between the Ivy API tests and the frontend tests, and we should strive for as much sharing as possible to minimize the amount of code.

**Running Ivy Frontend Tests**

The CI Pipeline runs the entire collection of Frontend Tests for the frontend that is being updated on every push to the repo.

You will need to make sure the Frontend Test is passing for each Ivy Frontend function you introduce/modify.
If a test fails on the CI, you can see details about the failure under `Details -> Run Frontend Tests` as shown in `CI Pipeline`_.

You can also run the tests locally before making a PR. See the relevant `setting up`_ section for instructions on how to do so.


**Round Up**

This should have hopefully given you a good understanding of Ivy Frontend Tests!

If you have any questions, please feel free to reach out on `discord`_ in the `ivy frontends tests channel`_ or in the `ivy frontends tests forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/iS7QFsQa9bI" class="video">
    </iframe>