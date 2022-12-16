.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true
   :width: 100%

.. raw:: html

    <br/>
    <div align="center">
    <a href="https://github.com/unifyai/ivy/issues">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/issues/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/network/members">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/forks/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/stargazers">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/stars/unifyai/ivy">
    </a>
    <a href="https://github.com/unifyai/ivy/pulls">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://pypi.org/project/ivy-core">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-core.svg">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/unifyai/ivy/docs?label=docs">
    </a>
    <a href="https://github.com/unifyai/ivy/actions?query=workflow%3Atest-ivy">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/unifyai/ivy/test-ivy?label=tests">
    </a>
    <a href="https://discord.gg/G4aR9Q7DTN">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    </div>
    <br clear="all" />
CI Dashboard
---------------------
We require a unit test table📄 which show the output of various tests related to each submodule and
corresponding backend. 

We solve that issue by maintaining a database📊, to which the job output is pushed right after the workflow runs. We then pull the data from the database🔑, do some wrangling in a script and push a table result for each of these submodules in this branch. The rows consist of each test for each module(functional/stateful), and the columns consist of each backend framework. 

The dashboard script is triggered every 20 mins and  is deployed on cloud. The `script <https://github.com/unifyai/ivy/blob/4695422471eb820e4d8f5146878deadd6ba47a44/automation_tools/dashboard_automation/update_db.py>`_  used for updating the database is added as a step into the action workflows.

Status
--------

👉To view the status of the tests at any given time, head on to -:

* `Array API Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/array_api_dashboard.md>`_
* `Functional Core Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/functional_core_dashboard.md>`_
* `Functional NN Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/functional_nn_dashboard.md>`_
* `Stateful Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/stateful_dashboard.md>`_
* `Experimental Core Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/experimental_core_dashboard.md>`_
* `Experimental NN Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/experimental_nn_dashboard.md>`_
* `Torch Frontend Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/torch_frontend_dashboard.md>`_
* `Jax Frontend Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/jax_frontend_dashboard.md>`_
* `Tensorflow Frontend Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/tf_frontend_dashboard.md>`_
* `Numpy Frontend Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/numpy_frontend_dashboard.md>`_
* `Miscellaneous Tests <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/misc_test_dashboard.md>`_

These are higher level Submodule specific dashboards, for more fine grained individual test dashboards click on the badges✅ inside these submodules.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
