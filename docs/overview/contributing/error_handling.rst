Error Handling
==============

.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`pycharm thread`: https://discord.com/channels/799879767196958751/1186628916522262629
.. _`docker thread`: https://discord.com/channels/799879767196958751/1186629067966009424
.. _`pre-commit thread`: https://discord.com/channels/799879767196958751/1186629635694399539
.. _`pip packages thread`: https://discord.com/channels/799879767196958751/1186629837515935765
.. _`ivy tests thread`: https://discord.com/channels/799879767196958751/1189907526226034698

This section, "Error Handling" aims to assist you in navigating through some common errors you might encounter while working with the Ivy's Functional API. We'll go through some common errors which you might encounter while working as a contributor or a developer.

#. This is the case where we pass in a dtype to `torch` which is not actually supported by the torch's native framework itself.

   .. code-block:: python

        E       RuntimeError: "logaddexp2_cpu" not implemented for 'Half'
        E       Falsifying example: test_logaddexp2(
        E           backend_fw='torch',
        E           on_device='cpu',
        E           dtype_and_x=(['float16', 'float16'],
        E            [array([-1.], dtype=float16), array([-1.], dtype=float16)]),
        E           test_flags=FunctionTestFlags(
        E               ground_truth_backend='tensorflow',
        E               num_positional_args=2,
        E               with_out=False,
        E               instance_method=False,
        E               test_gradients=False,
        E               test_trace=None,
        E               as_variable=[False],
        E               native_arrays=[False],
        E               container=[False],
        E           ),
        E           fn_name='logaddexp2',
        E       )
        E
        E       You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXicY2BkAAMoBaaR2WAAAACVAAY=') as a decorator on your test case


   **Solution:**

   As we are explicitly passing in a `dtype` which is not supported in the torch framework itself so torch backend fails here, a possible fix is adding the dtype in the unsupported dtype         decoartor which would look something like this.

   .. code-block:: python

        @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)

   and place it above the function definition.

#. This is the case where the value from the ground-truth backend(tensorflow) does not match the value of the backend(jax) we are testing for this case.

   .. code-block:: python

        E       AssertionError:  the results from backend jax and ground truth framework tensorflow do not match
        E        0.25830078125!=0.258544921875
        E
        E
        E       Falsifying example: test_acosh(
        E           backend_fw='jax',
        E           on_device='cpu',
        E           dtype_and_x=(['float16'], [array(4., dtype=float16)]),
        E           test_flags=FunctionTestFlags(
        E               ground_truth_backend='tensorflow',
        E               num_positional_args=1,
        E               with_out=False,
        E               instance_method=False,
        E               test_gradients=True,
        E               test_trace=None,
        E               as_variable=[False],
        E               native_arrays=[False],
        E               container=[False],
        E           ),
        E           fn_name='acosh',
        E       )
        E
        E       You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXicY2BAABYQwQgiAABDAAY=') as a decorator on your test case

   **Solution:**

   As both the results are pretty close to each others in this case, adding an `rtol = 10^-3` and `atol = 10^-3` would fix the failing tests here.

         .. code-block:: python

               @handle_test(
                   fn_tree="functional.ivy.acosh",
                   dtype_and_x=helpers.dtype_and_values(
                       available_dtypes=helpers.get_dtypes("float"),
                       min_value=1,
                       large_abs_safety_factor=4,
                       small_abs_safety_factor=4,
                   ),
               )
               def test_acosh(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
                   input_dtype, x = dtype_and_x
                   helpers.test_function(
                       input_dtypes=input_dtype,
                       test_flags=test_flags,
                       backend_to_test=backend_fw,
                       fn_name=fn_name,
                       on_device=on_device,
                       rtol_=1e-2,
                       atol_=1e-2,
                       x=x[0],
                   )

#. This is a similar assertion as stated in point 2 but with torch and ground-truth tensorflow not matching but the matrices are quite different so there should be an issue in the backends rather than a numerical instability here.

   .. code-block:: python

        E       AssertionError:  the results from backend torch and ground truth framework tensorflow do not match
        E        [[1.41421356 1.41421356 1.41421356]
        E        [1.41421356 1.41421356 1.41421356]
        E        [1.41421356        inf 1.41421356]]!=[[1.41421356e+000 1.41421356e+000 1.41421356e+000]
        E        [1.41421356e+000 1.41421356e+000 1.41421356e+000]
        E        [1.41421356e+000 1.34078079e+154 1.41421356e+000]]
        E
        E
        E       Falsifying example: test_abs(
        E           backend_fw='torch',
        E           on_device='cpu',
        E           dtype_and_x=(['complex128'],
        E            [array([[-1.-1.00000000e+000j, -1.-1.00000000e+000j, -1.-1.00000000e+000j],
        E                    [-1.-1.00000000e+000j, -1.-1.00000000e+000j, -1.-1.00000000e+000j],
        E                    [-1.-1.00000000e+000j, -1.-1.34078079e+154j, -1.-1.00000000e+000j]])]),
        E           fn_name='abs',
        E           test_flags=FunctionTestFlags(
        E               ground_truth_backend='tensorflow',
        E               num_positional_args=1,
        E               with_out=False,
        E               instance_method=False,
        E               test_gradients=False,
        E               test_trace=None,
        E               as_variable=[False],
        E               native_arrays=[False],
        E               container=[False],
        E           ),
        E       )
        E
        E       You can reproduce this example by temporarily adding @reproduce_failure('6.82.4', b'AXicY2ZkYAIiBiBgZIAAxqHEXsAAB7jUQAAAMtEAzQ==') as a decorator on your test case

   **Solution:**

   If this is passing for all other backends and just failing for torch, and the result matrices are also different which states there is not a numerical instability, the issue is with the       torch backend. The best approach in this case is to see the torch backend, there should be an issue in the implementation. You have to correct the backend implementation for torch.

**Note**

This section is specifically targeted towards dealing with the Ivy Functional API and the Ivy Experimental API.

**Round Up**

This should have hopefully given you an understanding of how to deal with common errors while working with the the functional API.

If you have any questions, please feel free to reach out on `discord`_  in the `ivy tests thread`_, `pycharm thread`_, `docker thread`_, `pre-commit thread`_, `pip packages thread`_ depending on the question!
