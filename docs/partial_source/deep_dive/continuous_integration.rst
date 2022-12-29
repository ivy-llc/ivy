Continuous Integration
======================

.. _`continuous integration channel`: https://discord.com/channels/799879767196958751/982737993028755496
.. _`continuous integration forum`: https://discord.com/channels/799879767196958751/982737993028755496
.. _`discord`: https://discord.gg/sXyFF8tDtm

We follow the practice of Continuous Integration (CI), in order to regularly build and test code at Ivy.
This makes sure that:

#. Developers get feedback on their code soon, and Errors in the Code are detected quickly. ✅
#. The developer can easily debug the code when finding the source of an error, and rollback changes in case of Issues. 🔍

In order to incorporate Continuous Integration in the Ivy Repository, we follow a three-fold technique, which involves:

#. Commit Triggered Testing
#. Periodic Testing
#. Manual Testing

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/continuous_integration/CI.png?raw=true
   :alt: CI Overview

We use GitHub Actions in order to implement and automate the process of testing. GitHub Actions allow implementing custom workflows that can build the code in the repository and run the tests. All the workflows used by Ivy are defined in the `.github/workflows <https://github.com/unifyai/ivy/tree/master/.github/workflows>`_ directory.

Commit (Push/PR) Triggered Testing
----------------------------------

The following Tests are triggered in case of a Commit (Push/PR) made to the Ivy Repository:

#. Ivy Tests (A small subset)
#. Array API Tests

Ivy Tests
---------
A test is defined as the triplet of (submodule, function, backend). We follow the following notation to identify each test:
:code:`submodule::function,backend`

For example, :code:`ivy_tests/test_ivy/test_frontends/test_torch/test_tensor.py::test_torch_instance_arctan_,numpy`

The Number of such Ivy tests running on the Repository (without taking any Framework/Python Versioning into account) is 7284 (as of writing this documentation), and we are adding tests daily. Therefore, triggering all the tests on each commit is neither desirable (as it will consume a huge lot of Compute Resources, as well take a large amount of time to run) nor feasible (as Each Job in Github Actions has a time Limit of 360 Minutes, and a Memory Limit as well).

Further, When we consider versioning, for a single Python version, and ~40 frontend and backend versions, the tests would shoot up to 40 * 40 * 7284 = 11,654,400, and we obviously don't have resources as well as time to run those many tests on each commit.

Thus, We need to prune the tests that run on each push to the Github Repository. The ideal situation, here, is to trigger only the tests that are impacted by the changes made in a push. The tests that are not impacted by the changes made in a push, are wasteful to trigger, as their results don’t change (keeping the same Hypothesis Configuration). For example, Consider the `commit <https://github.com/unifyai/ivy/commit/29cc90dda9e9a8d64789ed28e6eab0f41257a435>`_

The commit changes the :code:`_reduce_loss` function and the :code:`binary_cross_entropy` functions in the ivy/functional/ivy/losses.py file. The only tests that must be triggered (for all 4 backends) are:

:code:`ivy_tests/test_ivy/test_functional/test_nn/test_losses.py::test_binary_cross_entropy_with_logits`
:code:`ivy_tests/test_ivy/test_functional/test_nn/test_losses.py::test_cross_entropy`
:code:`ivy_tests/test_ivy/test_functional/test_nn/test_losses.py::test_binary_cross_entropy`
:code:`ivy_tests/test_ivy/test_functional/test_nn/test_losses.py::test_sparse_cross_entropy`
:code:`ivy_tests/test_ivy/test_frontends/test_torch/test_loss_functions.py::test_torch_binary_cross_entropy`
:code:`ivy_tests/test_ivy/test_frontends/test_torch/test_loss_functions.py::test_torch_cross_entropy`

Ivy’s Functional API functions :code:`binary_cross_entropy_with_logits`, :code:`test_cross_entropy`, :code:`test_binary_cross_entropy`, :code:`test_sparse_cross_entropy`, are precisely the ones impacted by the changes in the commit, and since the torch Frontend Functions torch_binary_cross_entropy, and torch_cross_entropy are wrapping these, the corresponding frontend tests are also impacted. No other Frontend function calls these underneath and hence are not triggered.

How do we (or at least try to) achieve this?

Implementation
A Top-Down View:
In order to implement this, we use the magic of Test Coverage!
Test Coverage refers to finding statements (lines) in your code that are executed (or could have been executed), on running a particular test. For example, (TODO: Give an example of Ivy Test and its coverage).

We use the Python Coverage Package (https://coverage.readthedocs.io/en/7.0.0/) for determining the Test Coverage of our tests.

The way it works is by running a particular pytest, and then logging each line (of our code) that was executed (or could have been executed) by the test.

Computing Test Coverage for all Ivy tests, allows us to find, for each line of code, which tests affect the same. We create a Dictionary (Mapping) to store this information as follows (The actual Mapping we prepare is a bit different from this design, but we will follow this for now due to Pedagogical Purposes):

.. math::

    \begin{flalign}
    \{ \\
     \ \ \ \ "f_1": [\{\}, \{"t_1","t_3","t_7"\}, …, \{"t_10","t_11","t_15"\}], \\
     \ \ \ \ … \\
     \ \ \ \ "f_m": [\{\}, \{"t_11","t_23","t_37"\}, …, \{"t_32","t_54","t_65"\}] \\
    \}
    \end{flalign}

The dictionary thus stores a list for each file f1 … fm. The list is a sequence encapsulating the lines of the file. Each index of the list contains a set of tests, which are mapped to the corresponding line in the file.

So Yeah, Given this Mapping for a commit, We can just follow the below procedure:
Find the files which are changed in the commit, and check for lines that are added/deleted/updated in the file.
Determine the Tests that impact the lines, and trigger just those tests, and no other.

But, there’s a fundamental issue here, Computing the Mapping requires determining the coverage for all tests, which involves running all the tests. Doesn’t this sound cyclical? After all, We are doing all this to avoid running all the tests.

Now assume that you had some way to update the Mapping for a commit from the previous Mapping without having to run all the tests. Then, Given the Mapping for a single commit, we could follow this to determine and run the relevant tests for each commit as follows:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/continuous_integration/ITRoadmap.png?raw=true
   :alt: Intelligent Testing Roadmap
This is exactly what we do in order to implement Intelligent Testing. The “Update Mapping” Logic works as follows for each changed file:

1. For each deleted line, we remove the corresponding entry from the list corresponding to the file in the Mapping.

.. code-block:: python

    tests_file = tests[file_name]
    for line in sorted(deleted, reverse=True):
       if line < len(tests_file):
           del tests_file[line]


2. For each line added, we compute the tests as an intersection of the set of tests on the line above and below the line.

.. code-block:: python

    for line in added:
       top = -1
       bottom = -1
       if 0 <= line - 1 < len(tests_file):
           top = tests_file[line - 1]
       if 0 <= line + 1 < len(tests_file):
           bottom = tests_file[line + 1]
       tests_line = set()
       if top != -1 and bottom != -1:
           tests_line = top.intersection(bottom)
       elif top != -1:
           tests_line = top
       elif bottom != -1:
           tests_line = bottom
       tests_file.insert(line, tests_line)
    tests[file_name] = tests_file


3. Finally, For newly added tests, we compute the coverage of the new tests (limited to 10 per commit), and update the Mapping correspondingly.

Once the Mapping has been updated, the “Determine & Run Tests” Logic works as follows:

1. For each deleted line, we collect the tests corresponding to the line as:

.. code-block:: python

    for line in deleted:
       tests_to_run = determine_tests_line(tests_file, line, tests_to_run)

2. For each line updated, we collect the tests corresponding to the line as:

.. code-block:: python

    for line in updated:
       tests_to_run = determine_tests_line(tests_file, line, tests_to_run)

3. For each line added, we collect the tests corresponding to the line as:

.. code-block:: python

    for line in added:
       tests_to_run = determine_tests_line(tests_file, line, tests_to_run)

4. Further, All the new tests added in a commit are collected (up to a max limit of 10, any more tests added are taken up in subsequent commits).
5. Finally, All the collected tests are triggered by the run_tests.py script, and the corresponding entry in the MongoDB Database is updated with the Test Result (Details on this in the Dashboard Section below).


Array API Tests
---------------
The `test-array-api.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-array-api.yml>`_ workflow runs the Array API Tests.
Other than being triggered on push and pull requests with the required labels, It can also be manually dispatched from the `Actions <https://github.com/unifyai/ivy/actions>`_ Tab.

The Workflow runs the Array API Tests for each backend and submodule pair.
More details about the Array API Tests are available `here <https://lets-unify.ai/ivy/deep_dive/array_api_tests.rst.html>`_.

Ivy Core Tests
--------------

The `test-ivy-core.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-core.yml>`_ Workflow runs the Ivy Core Tests.

Individual Tests in the Workflow are triggered only on changes to specific files.
For a given backend :code:`b` and submodule :code:`s`, the corresponding test is run only if the commit changes the following files (and otherwise, it is skipped):

#. :code:`ivy_tests/test_ivy/test_functional/test_core/test_s.py`
#. :code:`ivy_tests/test_ivy/helpers.py`
#. :code:`ivy/array/s.py`
#. :code:`ivy/container/s.py`
#. :code:`ivy/functional/backends/b/s.py`
#. :code:`ivy/functional/ivy/s.py`

In case you want to run all the Ivy Core Tests, a manually-triggered workflow is available `here <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-core-manual.yml>`_ that can be dispatched from the `Actions <https://github.com/unifyai/ivy/actions>`_ tab.

More details about Ivy Tests are available `here <https://lets-unify.ai/ivy/deep_dive/ivy_tests.html>`_.

Ivy NN Tests
------------

The `test-ivy-core.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-nn.yml>`_ workflow runs the Ivy NN Tests.

Similar to the Ivy Core Tests Workflow, Individual Tests are triggered only on changes to specific files.
For a given backend :code:`b` and submodule :code:`s`, the test is run only if the commit changes the following files (and otherwise, it is skipped):

#. :code:`ivy_tests/test_ivy/test_functional/test_nn/test_s.py`
#. :code:`ivy_tests/test_ivy/helpers.py`
#. :code:`ivy/array/s.py`
#. :code:`ivy/container/s.py`
#. :code:`ivy/functional/backends/b/s.py`
#. :code:`ivy/functional/ivy/s.py`

Similar to the Ivy Core Tests Workflow, in case you want to run all the Ivy NN Tests, a manually-triggered workflow is available `here <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-nn-manual.yml>`_.


Ivy Stateful Tests
------------------
The `test-ivy-stateful.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-stateful.yml>`_ workflow runs the Ivy Stateful Tests.

In this case too, Individual Tests are triggered only on changes to specific files.
For a given backend :code:`b` and submodule :code:`s`, the test is run only if the commit changes the following files (and otherwise, it is skipped):

#. :code:`ivy_tests/test_ivy/test_stateful/test_s.py`
#. :code:`ivy_tests/test_ivy/helpers.py`
#. :code:`ivy/array/s.py`
#. :code:`ivy/container/s.py`
#. :code:`ivy/functional/backends/b/s.py`
#. :code:`ivy/functional/ivy/s.py`
#. :code:`ivy/stateful/s.py`

Similar to the Ivy Core Tests Workflow, in case you want to run all the Ivy Stateful Tests, there is a manually-triggered workflow available `here <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-stateful-manual.yml>`_.

Ivy Frontend Tests
------------------
The following workflows run the Frontend tests for the corresponding backend:

#. **Jax**: `test-frontend-jax.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-frontend-jax.yml>`_
#. **NumPy**: `test-frontend-numpy.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-frontend-numpy.yml>`_
#. **TensorFlow**: `test-frontend-tensorflow.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-frontend-tensorflow.yml>`_
#. **PyTorch**: `test-frontend-torch.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-frontend-torch.yml>`_

Each of these workflows can also be Manually dispatched from the `Actions <https://github.com/unifyai/ivy/actions>`_ Tab.
More details about the Array API Tests are available `here <https://lets-unify.ai/ivy/deep_dive/ivy_frontends_tests.html>`_.


CI Pipeline ➡️
-------------
The below subsections provide the roadmap for running workflows and interpreting results in case a push or a pull request is made to the repository.

Push
^^^^
Whenever a push is made to the repository, a variety of workflows are triggered automatically (as described above).
This can be seen on the GitHub Repository Page, with the commit message followed by a yellow dot, indicating that some workflows have been queued to run following this commit, as shown below:


.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/continuous_integration/push.png?raw=true
   :alt: Push

Clicking on the yellow dot (🟡) (which changes to a tick (✔) or cross (❌), when the tests have been completed) yields a view of the test-suite results as shown below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/continuous_integration/push-2.png?raw=true
   :alt: Test-Suite

Click on the "Details" link corresponding to the failing tests, in order to identify the cause of the failure.
It redirects to the Actions Tab, showing details of the failure, as shown below:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/continuous_integration/push-3.png?raw=true
   :alt: Workflow Result

Click on the corresponding section, as given below, in order to see the logs of the failing tests:

#. **Array API Tests**: Run Array Api Tests
#. **Ivy Core Tests**: Run Functional-Core Tests
#. **Ivy NN Tests**: Run Functional-NN Tests
#. **Ivy Stateful Tests**: Run Stateful Tests
#. **Ivy Frontend Tests**: Run Frontend Test

You can ignore the other sections of the Workflow, as they are for book-keeping and implementation purposes.

Pull Request
^^^^^^^^^^^^
In case of a pull request, the test suite is available on the Pull Request Page on Github, as shown below:


.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/deep_dive/continuous_integration/pull-request1.png?raw=true
   :alt: PR Test-Suite

Clicking on the "Details" link redirects to the Action Log.
The rest of the procedure remains the same as given in the Push section above.

Scheduled Tests (Cron Jobs)
---------------------------

In order to make sure that no tests are ignored for a long time, as well as, decouple the commit frequency with the testing frequency, we use Scheduled Tests (Cron Jobs) to run an Ivy Core, Ivy NN, and Ivy Stateful Test every hour
The following workflows run cron jobs:

#. `test-ivy-core-cron.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-core-cron.yml>`_

#. `test-ivy-nn-cron.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-nn-cron.yml>`_

#. `test-ivy-stateful-cron.yml <https://github.com/unifyai/ivy/blob/master/.github/workflows/test-ivy-stateful-cron.yml>`_

The cron jobs are used to update the latest results in the Dashboard, as explained in the following section.

Dashboard
---------
In order to view the status of the tests, at any point in time, we maintain a dashboard containing the results of the latest Workflow that ran each test.
These are the links to the Dashboard for the given workflows:

#. `Array API Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/array_api_dashboard.md>`_
#. `Ivy Core Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/functional_core_dashboard.md>`_
#. `Ivy NN Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/functional_nn_dashboard.md>`_
#. `Ivy Stateful Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/stateful_dashboard.md>`_

The status badges are clickable, and will take you directly to the Action log of the latest workflow that ran the corresponding test.

**Round Up**

This should have hopefully given you a good feel for how function wrapping is applied to functions in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `continuous integration channel`_
or in the `continuous integration forum`_!