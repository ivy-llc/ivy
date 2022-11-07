Ivy Frontend Tests
==================

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html
.. _`ivy frontends tests channel`: https://discord.com/channels/799879767196958751/1028267758028337193
.. _`ivy frontends tests forum`: https://discord.com/channels/799879767196958751/1028297887605587998
.. _`test ivy`: https://github.com/unifyai/ivy/tree/db9a22d96efd3820fb289e9997eb41dda6570868/ivy_tests/test_ivy
.. _`test_frontend_function`: https://github.com/unifyai/ivy/blob/591ac37a664ebdf2ca50a5b0751a3a54ee9d5934/ivy_tests/test_ivy/helpers.py#L1047
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`Function Wrapping`: https://lets-unify.ai/ivy/deep_dive/3_function_wrapping.html
.. _`open task`: https://lets-unify.ai/ivy/contributing/open_tasks.html#open-tasks
.. _`Ivy Tests`: https://lets-unify.ai/ivy/deep_dive/ivy_tests.html
.. _`Function Testing Helpers`: https://github.com/unifyai/ivy/blob/bf0becd459004ae6cffeb3c38c02c94eab5b7721/ivy_tests/test_ivy/helpers/function_testing.py

Introduction
------------

Just like the backend functional API, our frontend functional API has a collection of Ivy tests located in subfolder `test ivy`_.
In this section of the deep dive we are going to jump into Ivy Frontend Tests!

**Writing Ivy Tests**

The Ivy tests in this section make use of hypothesis for performing property based testing which is documented in detail in the Ivy Tests section of the Deep Dive.
We assume knowledge of hypothesis data generation strategies and how to implement them for testing.

**Important Helper Functions**

* :func:`handle_cmd_line_args` a decorator that should be added to every test function.
  For more information, visit the `Function Wrapping`_ section of the docs.

* :func:`helpers.test_frontend_function` helper function that is designed to do the heavy lifting and make testing Ivy Frontends easy!
  One of the many `Function Testing Helpers`_.
  It is used to test a frontend function for the current backend by comparing the result with the function in the associated framework.

* :func:`helpers.get_dtypes` helper function that returns either a full list of data types or a single data type, we should **always** be using `helpers.get_dtypes` to sample data types.

* :func:`helpers.dtype_and_values` is a convenience function that allows you to generate arrays of any dimension and their associated data types, returned as :code:`([dtypes], [np.array])`.

* :func:`helpers.num_positional_args` is a convenience function that specifies the number of positional arguments for a particular function.

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
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.lax.tan"
        ),
    )
    def test_jax_lax_tan(
        dtype_and_x,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            fn_tree="lax.tan",
            x=x[0],
        )

* As you can see we generate almost everything we need to test a frontend function within the :code:`@given` and :code:`@handle_cmd_line_args` decorators.
* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid :code:`float` data types specifically for Jax.
* We pass :code:`fn_name` to :func:`helpers.num_positional_args` which is used to determine the number of positional arguments for :code:`jax.lax.tan`.
* We do not generate any values for :code:`fw`, these values are generated by :func:`handle_cmd_line_args` and are only passed as an argument to :func:`test_jax_lax_tan`.
* We unpack the :code:`dtype_and_x` to :code:`input_dtype` and :code:`x`.
* We then pass the generated values to :code:`helpers.test_frontend_function` which tests the frontend function.
* We set :code:`fn_tree` to :code:`lax.tan` which is the path to the function in the Jax namespace.
* :func:`jax.lax.tan` does not support :code:`out` arguments so we set :code:`with_out` to :code:`False`.
* One last important note is that all helper functions are designed to take keyword arguments only.

**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_mathematical_functions/test_np_trigonometric_functions.py
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric")
        ),
        dtype=helpers.get_dtypes("float", full=False, none=True),
        where=np_frontend_helpers.where(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.tan"
        ),
    )
    def test_numpy_tan(
        dtype_and_x,
        dtype,
        where,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
            where=where,
            input_dtype=input_dtype,
            as_variable=as_variable,
            native_array=native_array,
        )
        np_frontend_helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="numpy",
            fn_tree="tan",
            x=x[0],
            where=where,
            dtype=dtype[0],
        )

* Here we use :code:`helpers.get_dtypes("numeric")` to generate :code:`available_dtypes`, these are valid :code:`numeric` data types specifically for NumPy.
* NumPy has an optional argument :code:`where` which is generated using :func:`np_frontend_helpers.where`.
* :func:`numpy.tan` supports :code:`out` arguments so we set generate values for :code:`with_out`.
* Using :func:`np_frontend_helpers.handle_where_and_array_bools` we do some processing on the generated :code:`where` value.
* Instead of :func:`helpers.test_frontend_function` we use :func:`np_frontend_helpers.test_frontend_function` which behaves the same but has some extra code to handle the :code:`where` argument.
* We set :code:`fn_tree` to :code:`tan` which is the path to the function in the NumPy namespace.
* :code:`casting`, :code:`order`, :code:`subok` and are other other optional arguments for :func:`numpy.tan`.

**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_math.py
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.tan"
        ),
    )
    def test_tensorflow_tan(
        dtype_and_x, as_variable, num_positional_args, native_array, fw
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="tensorflow",
            fn_tree="tan",
            x=x[0],
        )

* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for TensorFlow.
* We set :code:`fn_tree` to :code:`tan` which is the path to the function in the TensorFlow namespace.


**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_non_linear_activation_functions.py
    # leaky_relu
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.torch.nn.functional.leaky_relu"
        ),
        alpha=st.floats(min_value=0, max_value=1),
        with_inplace=st.booleans(),
    )
    def test_torch_leaky_relu(
        dtype_and_x,
        with_out,
        with_inplace, # does handle_cmd_line_args deals with this like with_out?
        num_positional_args,
        as_variable,
        native_array,
        fw,
        alpha,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            with_inplace=with_inplace,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            fn_tree="nn.functional.leaky_relu",
            input=x[0],
            negative_slope=alpha,
        )

* We use :code:`helpers.get_dtypes("float")` to generate :code:`available_dtypes`, these are valid float data types specifically for PyTorch.
* We set :code:`fn_tree` to :code:`nn.functional.leaky_relu` which is the path to the function in the PyTorch namespace.
* We get :code:`with_inplace` with hypothesis to test the function that supports direct inplace update in its arguments: when :code:`with_inplace` is :code:`True` the function updates the :code:`input` argument with return value and the return value has the same reference as the input.
* We should set :code:`with_inplace` is :code:`True` for the special In-place versions of PyTorch functions that always do inplace update, as the :code:`input` argument is also updated with return value and the returned value has the same reference as the input.

ivy.full()
^^^^^^^^^^

Here we are going to look at an example of a function that does not consume an :code:`array`.
This is the creation
function :func:`full`, which takes an array shape as an argument to create an array and filled with elements of a given value.
This function requires us to create extra functions for generating :code:`shape` and :code:`fill value`, these use the :code:`shared` hypothesis strategy.


**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_lax_operators.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))


    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.lax.full"
        ),
    )
    def test_jax_lax_full(
        shape,
        fill_value,
        dtypes,
        native_array,
        as_variable,
        num_positional_args,
        as_variable,
        native_array,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            fn_tree="lax.full",
            shape=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
        )

* The custom function we use is :code:`_fill_value` which generates a :code:`fill_value` to use for the :code:`fill_value` argument but handles the complications of :code:`int` and :code:`uint` types correctly.
* We use the helper function :func:`helpers.get_shape` to generate :code:`shape`.
* We use :code:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Jax.
  This is used to specify the data type of the output array.
* :func:`full` does not consume :code:`array`, we set :code:`as_variable_flags`, :code:`native_array_flags` to :code:`[False]` and :code:`with_out` :code:`False`.


**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/creation_routines/test_from_shape_or_value.py
    @st.composite
    def _fill_value(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_uint_dtype(dtype):
            return draw(helpers.ints(min_value=0, max_value=5))
        if ivy.is_int_dtype(dtype):
            return draw(helpers.ints(min_value=-5, max_value=5))
        return draw(helpers.floats(min_value=-5, max_value=5))

    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.numpy.full"
        ),
    )
    def test_numpy_full(
        shape,
        fill_value,
        dtypes,
        as_variable,
        native_array,
        num_positional_args,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="numpy",
            fn_tree="full",
            shape=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
        )

* We use :func:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for NumPy.
* :func:`numpy.full` does not have a :code:`where` argument so we can use :func:`helpers.test_frontend_function`

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

    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.fill"
        ),
    )
    def test_tensorflow_full(
        shape,
        fill_value,
        dtypes,
        as_variable,
        native_array,
        num_positional_args,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="tensorflow",
            fn_tree="fill",
            dims=shape,
            value=fill_value,
            rtol=1e-05,
        )

* We use :func:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for TensorFlow.
* Tensorflow's version of :func:`full` is named :func:`fill` therefore we specify the :code:`fn_tree` argument to be :code:`"fill"`
* When running the test there where some small discrepancies between the values so we can use :code:`rtol` to specify the relative tolerance.


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


    @st.composite
    def _requires_grad(draw):
        dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
        if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
            return draw(st.just(False))
        else:
            return draw(st.booleans())


    @handle_cmd_line_args
    @given(
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
        fill_value=_fill_value(),
        dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
        requires_grad=_requires_grad(),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.torch.full"
        ),
    )
    def test_torch_full(
        shape,
        fill_value,
        dtypes,
        requires_grad,
        device,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        helpers.test_frontend_function(
            input_dtypes=dtypes,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            fn_tree="full",
            size=shape,
            fill_value=fill_value,
            dtype=dtypes[0],
            device=device,
            requires_grad=requires_grad,
        )

* Here we created another extra function, :code:`_requires_grad()`, to accommodate the :code:`requires_grad` argument.
  This is because when the dtype is an integer or unsigned integer the :code:`requires_grad` argument is not supported.
* We use :code:`helpers.get_dtypes` to generate :code:`dtype`, these are valid numeric data types specifically for Torch.
* :func:`torch.full` supports :code:`out` so we generate :code:`with_out`.

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
    @handle_cmd_line_args
    @given(
        dtype_and_inputs=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=2,
            allow_inf=False,
            shared_dtype=True,
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.torch.greater"
        ),
    )
    def test_torch_greater(
        dtype_and_inputs,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
    ):
        input_dtype, inputs = dtype_and_inputs
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            has_aliases=True,
            all_aliases=["gt"],
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            frontend="torch",
            fn_tree="greater",
            input=inputs[0],
            other=inputs[1],
            out=None,
        )

* We added a list of all aliases to the :code:`greater` function with a full namespace path such that when we are testing the original function we will test for the alias as well.
* During the frontend implementation, if a new alias is introduced you only need to go to the test function of the original frontend function and add that alias to :code:`all_aliases` argument in the :func:`test_frontend_function` helper with its full namespace.

Frontend Instance Method Tests
------------------------------

The frontend instance method tests are similar to the frontend function test, but instead of testing the function directly we test the instance method of the frontend class.

**Important Helper Functions**

:func:`helpers.test_frontend_instance_method` is used to test frontend instance methods.
It is used in the same way as :func:`helpers.test_frontend_function`.

**Useful Notes**
The :func:`helpers.test_frontend_instance_method` takes an argument :code:`frontend_class` which is the frontend class to test.
This is the relevant Ivy frontend class and not the native framework class.


Frontend Instance Method Test Examples
--------------------------------------

ivy.add()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_devicearray.py
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric", full=True),
            num_arrays=2,
            shared_dtype=True,
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.jax.DeviceArray.add",
        ),
    )
    def test_jax_instance_add(
        dtype_and_x,
        as_variable,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="jax",
            frontend_class=DeviceArray,
            fn_tree="DeviceArray.add",
            self=x[0],
            other=x[1],
        )

* We use :func:`test_frontend_array_instance_method` to test the instance method.
* We import the frontend class :class:`DeviceArray` from :code:`frontends.jax.DeviceArray` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`DeviceArray.add` which is the path to the function in the frontend class.
    
**NumPy**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_numpy/test_ndarray.py
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
        ),
    )
    def test_numpy_ndarray_add(
        dtype_and_x,
        as_variable,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_method(
            input_dtypes_init=input_dtype,
            as_variable_flags_init=as_variable,
            num_positional_args_init=0,
            native_array_flags_init=native_array,
            all_as_kwargs_np_init={
                "data": x[0],
            },
            input_dtypes_method=[input_dtype[1]],
            as_variable_flags_method=as_variable,
            num_positional_args_method=0,
            native_array_flags_method=native_array,
            all_as_kwargs_np_method={
                "value": x[1],
            },
            fw=fw,
            frontend="numpy",
            class_name="ndarray",
            method_name="add",
        )

* We use :func:`np_frontend_helpers.test_frontend_array_instance_method` to test the instance method.
  This handles the :code:`where` argument.
* We import the frontend class :class:`ndarray` from :code:`frontends.numpy.ndarray` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`ndarray.add` which is the path to the function in the frontend class.
    
**TensorFlow**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_tensorflow/test_tensor.py
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            shared_dtype=True,
        ),
        num_positional_args=helpers.num_positional_args(
            fn_name="ivy.functional.frontends.tensorflow.Tensor.add",
        ),
    )
    def test_tensorflow_instance_add(
        dtype_and_x, as_variable, num_positional_args, native_array, fw
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=False,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="tensorflow",
            frontend_class=Tensor,
            fn_tree="Tensor.add",
            self=x[0],
            y=x[1],
        )

* We import the frontend class :class:`Tensor` from :code:`frontends.tensorflow.tensor` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`Tensor.add` which is the path to the function in the frontend class.

**PyTorch**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_torch/test_tensor.py
    @handle_cmd_line_args
    @given(
        dtype_and_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            min_value=-1e04,
            max_value=1e04,
            allow_inf=False,
        ),
        alpha=st.floats(min_value=-1e06, max_value=1e06, allow_infinity=False),
        num_positional_args=helpers.num_positional_args(
            fn_name="functional.frontends.torch.Tensor.add",
        ),
    )
    def test_torch_instance_add(
        dtype_and_x,
        alpha,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
    ):
        input_dtype, x = dtype_and_x
        helpers.test_frontend_array_instance_method(
            input_dtypes=input_dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            fw=fw,
            frontend="torch",
            frontend_class=Tensor,
            fn_tree="Tensor.add",
            rtol=1e-04,
            self=x[0],
            other=x[1],
            alpha=alpha,
            out=None,
        )

* We import the frontend class :class:`Tensor` from :code:`frontends.torch.tensor` and pass it to the :code:`frontend_class` argument.
* We specify the :code:`fn_tree` to be :meth:`Tensor.add` which is the path to the function in the frontend class.


Frontend Special Method Tests
-----------------------------

The implementation for the frontend special method tests are somewhat a little different from how the instance methods are being tested.

**Important Helper Function**

:func:`helpers.value_test` is being used to test frontend special methods.


Frontend Special Method Test Examples
-------------------------------------

ivy.add()
^^^^^^^^^

**Jax**

.. code-block:: python

    # ivy_tests/test_ivy/test_frontends/test_jax/test_jax_devicearray.py
    @handle_cmd_line_args
    @given(
        dtype_x=helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric", full=True),
            shared_dtype=True,
            num_arrays=2,
        )
    )
    def test_jax_special_add(
        dtype_x,
        fw,
    ):
        input_dtype, x = dtype_x
        ret = DeviceArray(x[0]) + DeviceArray(x[1])
        ret_gt = jnp.array(x[0]) + jnp.array(x[1], dtype=input_dtype[1])
        ret = helpers.flatten_and_to_np(ret=ret)
        ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
        for (u, v) in zip(ret, ret_gt):
            helpers.value_test(
                ret=u,
                ret_from_gt=v,
                ground_truth_backend="jax",
            )

* We use :func:`helpers.value_test` to test the special method.
* We use the frontend class :class:`DeviceArray` to calculate jax frontend special method's result, which is then compared to the regular frontend function's result, when passed into the :func:`helpers.value_test`.
* We use :func:`helpers.value_test`,which takes an argument :code:`ground_truth_backend` which is the frontend that is to be tested.


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


**Round Up**

These examples have hopefully given you a good understanding of Ivy Frontend Tests!

If you have any questions, please feel free to reach out on `discord`_ in the `ivy frontends tests channel`_ or in the `ivy frontends tests forum`_!
