Fix Failing Tests:
==============================

.. _`repo`: https://github.com/unifyai/ivy
.. _`issues`: https://github.com/unifyai/ivy/issues?q=is%3Aopen+is%3Aissue+label%3A%22Failing+Test%22
.. _`issue`: https://github.com/unifyai/ivy/issues/25849
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`docker thread`: https://discord.com/channels/799879767196958751/1186629067966009424
.. _`miniconda`: https://docs.conda.io/en/latest/miniconda.html
.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`ivy/scripts/shell`: https://github.com/unifyai/ivy/tree/f71a414417646e1dfecb5de27fb555f80333932c/scripts/shell
.. _`platform compatibility tags`: https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
.. _`logging level`: https://docs.python.org/3/library/logging.html#logging.Logger.setLevel
.. _`pycharm thread`: https://discord.com/channels/799879767196958751/1186628916522262629
.. _`pre-commit thread`: https://discord.com/channels/799879767196958751/1186629635694399539
.. _`pip packages thread`: https://discord.com/channels/799879767196958751/1186629837515935765
.. _`ivy tests thread`: https://discord.com/channels/799879767196958751/1189907526226034698
.. _`ivy frontend tests thread`: https://discord.com/channels/799879767196958751/1190246804940402738

We're really happy you'd like to learn how to contribute towards Ivy 🙂

This page explains the main steps to get started with fixing failing tests!

Prerequirement:
**************************

Before you start with this you should have:

#. `Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
#. `Visual Studio Code here <https://code.visualstudio.com/>`_
#. `Docker Desktop <https://www.docker.com/products/docker-desktop>`_


Setting Up
***********

**Forking and cloning the repo**

#. `Fork Ivy Repo <https://github.com/unifyai/ivy/fork>`_
#. `Clone <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_ the fork with it's submoodules locally or on codespaces

   .. dropdown:: If you are new to Git:

      Depending on your preferred mode of cloning, any of the below should work:

      .. code-block:: bash

         git clone --recurse-submodules git@github.com:YOUR_USERNAME/ivy.git

      .. code-block:: bash

         git clone --recurse-submodules https://github.com/YOUR_USERNAME/ivy.git

      .. code-block:: bash

         gh repo clone YOUR_USERNAME/ivy your_folder -- --recurse-submodules

      Then enter into your cloned ivy folder, for example :code:`cd ~/ivy` and add Ivy original repository as upstream, to easily sync with the latest changes.

      .. code-block:: bash

         git remote add upstream https://github.com/unifyai/ivy.git

.. dropdown:: **Windows, docker and VsCode**

   #. Open the Docker desktop, make sure it's running in the background while following the process below.
   #. Open Ivy repo folder with Visual Studio Code, and follow the next steps:
      a. At the bottom right a window will pop up asking for "Dev Containers" extension, install that.
         In case the window doesn't pop up, search for the "Dev Containers" extension in the Visual Studio Code and install that.
      b. Install the "Docker" extension for Visual Studio Code, you'll easily find that by searching "docker" in the extensions tab.
      c. Once done, restart Visual Studio Code, at the bottom left corner there would be an icon similar to " >< " overlapped on each other.
      d. Clicking on that will open a bar at the top which will give you an option "Open Folder in Container...", click on that.
      e. Run tests with the next command "pytest test_file_path::test_fn_name". You are inside the container now, and you can locally run the tests that you've modified.

   .. warning::
      Opening the container may take a long time, as the Docker image is very large (5+ GB).


How to run tests
****************
To find tests which are currently failing, open the `issues`_ in our GitHub.,

You can notice :code:`test_jax_transpose` is failing in this `issue`_, this function is in the Jax frontends in the manipulaiton submodule.

To run test locally, you need to run the following command:

:code:`pytest test_file_path::test_fn_name`

In the case of :code:`test_jax_transpose`, the command will be

.. code-block:: bash

   pytest ivy_tests/test_ivy/test_frontends/test_jax/test_numpy/test_manipulations.py::test_jax_transpose

You will need to read through the errors in the terminal and use the common errors in the list at the end of this page to solve the test.

.. dropdown:: **Setting Up Testing for VsCode**

   The steps are as following to setup testing on VS Code.

   1. In the left toolbar menu, click on the flask Icon and select "Configure Python Tests" and select PyTest as the test framework.

   .. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/contributing/setting_up/vs_code_testing_setup/vs_testing_01.png?raw=true
      :width: 420

   1. Select ivy_tests as the root directory for testing.

   .. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/contributing/setting_up/vs_code_testing_setup/vs_testing_02.png?raw=true
      :width: 420

   1. Configure the _array_module.py file in the array_api_tests to be set to one of the supported frameworks.

   .. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/contributing/setting_up/vs_code_testing_setup/vs_testing_03.png?raw=true
      :width: 420

   1. Following all of this, you should refresh the test suite and you should now be able to run tests right from VS Code!

   2. To simply run the tests using the play button in the toolbar, you will need to add the .vscode folder to your workspace. Then add the ``settings.json`` file containing the following:

   .. code-block:: json

         {
            "python.testing.pytestArgs": [
               "./ivy_tests/test_ivy/",
               "./ivy_tests/array_api_testing/test_array_api/",
               "--continue-on-collection-errors",
            ],
            "python.testing.unittestEnabled": false,
            "python.testing.pytestEnabled": true,
            "python.testing.autoTestDiscoverOnSaveEnabled": true,
         }

Common Errors
*************

This section aims to assist you in navigating through some common errors you might encounter while working with the Ivy's Functional API. We'll go through :code:`test_jax_transpose` and then some common errors which you might encounter while working as a contributor or a developer.

#. Starting off with :code:`test_jax_transpose`, it throws an Assertion error because the shape returned by ground truth is different from the shape returned by the target backend.

   .. code-block:: python

    E       ivy.utils.exceptions.IvyBackendException: paddle: to_numpy: paddle: default_device: paddle: dev: (PreconditionNotMet) Tensor not initialized yet when DenseTensor::place() is called.
    E         [Hint: holder_ should not be null.] (at /paddle/paddle/phi/core/dense_tensor_impl.cc:61)
    E
    E       Falsifying example: test_jax_transpose(
    E           on_device='cpu',
    E           frontend='jax',
    E           backend_fw='paddle',
    E           array_and_axes=(array([], shape=(1, 0), dtype=complex64),
    E            ['complex64'],
    E            None),
    E           test_flags=FrontendFunctionTestFlags(
    E               num_positional_args=0,
    E               with_out=False,
    E               inplace=False,
    E               as_variable=[False],
    E               native_arrays=[False],
    E               test_trace=False,
    E               generate_frontend_arrays=False,
    E               transpile=False,
    E               precision_mode=True,
    E           ),
    E           fn_tree='ivy.functional.frontends.jax.numpy.transpose',
    E       )
    E
    E       You can reproduce this example by temporarily adding @reproduce_failure('6.87.3', b'AAEGBAEGAQAAAAAAAAAAAAAB') as a decorator on your test case

   **Solution:**

   As it is failing for torch backend and its producing a different shape than the ground truth, it is most likely a bug in the :code:`permute_dims` in torch backend which is being used in this frontend function.

   Now lets explore some other common errors you might face.

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


Where to ask for Help
*********************

The best place to ask for help is our `discord`_ server in the relevant channels. For instance, lets say you're facing an issue with :code:`test_jax_transpose` function, in this case you should post your query in the `ivy frontend tests thread`_.
