Fix Failing Tests
==========

.. _`repo`: https://github.com/unifyai/ivy
.. _`issues`: https://github.com/unifyai/ivy/issues
.. _`issue`: https://github.com/unifyai/ivy/issues/25849
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`pycharm channel`: https://discord.com/channels/799879767196958751/942114831039856730
.. _`docker channel`: https://discord.com/channels/799879767196958751/942114744691740772
.. _`pip packages channel`: https://discord.com/channels/799879767196958751/942114789642080317
.. _`miniconda`: https://docs.conda.io/en/latest/miniconda.html
.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`ivy/run_tests_CLI`: https://github.com/unifyai/ivy/tree/f71a414417646e1dfecb5de27fb555f80333932c/run_tests_CLI
.. _`platform compatibility tags`: https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
.. _`logging level`: https://docs.python.org/3/library/logging.html#logging.Logger.setLevel

We're really happy you'd like to learn how to contribute towards Ivy 🙂

This page explains the main steps to get started with fixing failing tests!

Setting Up
***********

**Forking and cloning the repo**


#. You will first need to fork the Ivy repository from the repository page here `repo`_ by using the fork button on the top right. This creates a copy of the Ivy repository in your GitHub account.
#. Clone your forked repo to your local machine.

Depending on your preferred mode of cloning, any of the below should work:

.. code-block:: none

    git clone --recurse-submodules git@github.com:YOUR_USERNAME/ivy.git

.. code-block:: none

    git clone --recurse-submodules https://github.com/YOUR_USERNAME/ivy.git

.. code-block:: none

    gh repo clone YOUR_USERNAME/ivy your_folder -- --recurse-submodules

Then enter into your cloned ivy folder, for example :code:`cd ~/ivy` and add Ivy original repository as upstream, to easily sync with the latest changes.

.. code-block:: none

    git remote add upstream https://github.com/unifyai/ivy.git

**Windows**

#. Install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_
#. Install `Visual Studio Code here <https://code.visualstudio.com/>`_
#. Open the Docker desktop, make sure it's running while following the process below.
   You can close the Docker desktop window afterwards, Docker will continue to run in the background.
#. Open Visual Studio Code, open the Ivy repo folder, and follow the steps listed below:

   a. At the bottom right a window will pop up asking for "Dev Containers" extension, install that.
      In case the window doesn't pop up, search for the "Dev Containers" extension in the Visual Studio Code and install that.
   b. Install the "Docker" extension for Visual Studio Code, you'll easily find that by searching "docker" in the extensions tab.
   c. Once done, restart Visual Studio Code, at the bottom left corner there would be an icon similar to " >< " overlapped on each other.
   d. Clicking on that will open a bar at the top which will give you an option "Open Folder in Container...", click on that.
   e. You'll be inside the container now, where you can locally run the tests that you've modified by running the command, "pytest test_file_path::test_fn_name". Opening the container may take a long time, as the Docker image is very large (5+ GB).


How to run tests
****************
To find tests which are currently failing, you can go to our github repo and go to `issues`_, then filter by label :code:`Failing Test`. 
There you will find all the failing tests
which you can work on. For instance :code:`test_jax_transpose` is failing in this `issue`_, this function is in the Jax frontends in the manipulaiton submodule.
To run tests locally, you can run the command "pytest test_file_path::test_fn_name". So in the case of :code:`test_jax_transpose`, the command will be 
:code:`pytest ivy/functional/frontends/jax/numpy/manipulations.py::test_jax_transpose`. You can also run tests via the green button but for that, you will first have to 
setup testing on vs code.

**Setting Up Testing**

The steps are as following to setup testing on VS Code.
1. Under the flask Icon in the toolbar select "Configure Python Tests" and select PyTest as the test framework.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/contributing/setting_up/vs_code_testing_setup/vs_testing_01.png?raw=true
   :width: 420

2. Select ivy_tests as the root directory for testing.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/contributing/setting_up/vs_code_testing_setup/vs_testing_02.png?raw=true
   :width: 420

3. Configure the _array_module.py file in the array_api_tests to be set to one of the supported frameworks.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/contributing/setting_up/vs_code_testing_setup/vs_testing_03.png?raw=true
   :width: 420

4. Following all of this, you should refresh the test suite and you should now be able to run tests right from VS Code!

5. To simply run the tests using the play button in the toolbar, you will need to add the .vscode folder to your workspace. Then add the ``settings.json`` file containing the following:

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

Note: Currently you do not need to comment out the :code:`conftest.py` file in the :code:`array_api_tests` directory.

