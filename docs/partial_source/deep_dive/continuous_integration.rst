Continuous Integration
======================

.. _`continuous integration channel`: https://discord.com/channels/799879767196958751/982737993028755496
.. _`continuous integration forum`: https://discord.com/channels/799879767196958751/982737993028755496
.. _`discord`: https://discord.gg/sXyFF8tDtm

We follow the practice of Continuous Integration (CI), in order to build and test our code at each commit (push or pull request) made to the Ivy Repository.
This makes sure that:

#. The commit doesn't introduce any new errors, in the code base. ✅
#. The developer can easily debug the code when finding the source of an error. 🔍

We use GitHub Actions in order to implement the CI Pipeline in our code, as explained in the following sections.

GitHub Actions
--------------

GitHub Actions allow implementing custom workflows that can build the code in the repository and run the tests.
Workflows can be configured to be triggered on the following events:

* **Push**: The Workflow triggers on a push to the repository with the possibility of getting started only on changes to specific files.
* **Pull Requests**: The Workflow triggers on any pull request made to the repository.
* **Schedule**: The Workflow runs on a fixed schedule, for ex., every hour, every day, etc.
  (Similar to Cron Jobs in Ubuntu).

All the workflows used by Ivy are defined in the .github/workflows directory.

The following sections describe the relevant Workflows used in the Ivy Repository, that implement the CI Pipeline.
Each of the workflows described below, are triggered on:

#. Any push made to the repository.
#. Any pull request made to the repository of the following types:

    * :code:`labeled`
    * :code:`opened`
    * :code:`synchronize`
    * :code:`reopened`
    * :code:`review_requested`

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