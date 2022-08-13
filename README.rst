.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/repos/ivy/logo.png?raw=true
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

**We’re on a mission to unify all ML frameworks 💥 + automate code conversions 🔄. pip install ivy-core 🚀, join our growing community 😊, and lets-unify.ai! 🦾**

.. raw:: html

    <div style="display: block;" align="center">
        <img width="3%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://mxnet.apache.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/mxnet_logo.png">
        </a>
        <img width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
    </div>

.. _docs: https://lets-unify.ai/ivy
.. _Colabs: https://drive.google.com/drive/folders/16Oeu25GrQsEJh8w2B0kSrD93w4cWjJAM?usp=sharing
.. _`contributor guide`: https://lets-unify.ai/ivy/contributing.html
.. _`open tasks`: https://lets-unify.ai/ivy/contributing/4_open_tasks.html

Overview
--------
Currently we are running the unit tests🧪 for all submodules(functional and stateful), for all backends
parallely in three Github Action workflow files using a :code:`matrix` strategy here -:

* `test-core <https://github.com/unifyai/ivy/blob/5da858be094a8ddb90ffe8886393c1043f4d8ae7/.github/workflows/test-ivy-core.yml>`_
* `test-nn   <https://github.com/unifyai/ivy/blob/5da858be094a8ddb90ffe8886393c1043f4d8ae7/.github/workflows/test-ivy-nn.yml>`_
* `test-stateful <https://github.com/unifyai/ivy/blob/5da858be094a8ddb90ffe8886393c1043f4d8ae7/.github/workflows/test-ivy-stateful.yml>`_

We require a unit test table📄 which show the output of various tests related to each submodule and
corresponding backend. Github Actions only allow a single badge per workflow, which is a shortcoming
since we are defining multiple jobs inside a workflow file using Grid Search across the matrix.

We solve that issue in this branch, by pulling data directly from the Github API and making a custom dashboard
for each of these submodules. The dashboard `script <https://github.com/unifyai/ivy/blob/23231be72dbfeb4d537769f48b9a077a687d98b3/automation_tools/dashboard_automation/dashboard_script.py>`_ is triggered every 20 mins by an action defined `here <https://github.com/unifyai/ivy/blob/23231be72dbfeb4d537769f48b9a077a687d98b3/.github/workflows/tests_dashboard.yml>`_.

The rows consist of each functional and stateful submodule, and the columns consist of each backend framework. There are
4 * 30 ~ 120 unique GitHub actions jobs in total, for running the Ivy tests. 

Status
--------

👉To view the status of the tests at any given time, head on to -:

* `Ivy Functional Core <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/functional_core_dashboard.md>`_
* `Ivy Functional NN <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/functional_nn_dashboard.md>`_
* `Ivy Stateful <https://github.com/unifyai/ivy/blob/dashboard/test_dashboards/stateful_dashboard.md>`_ 

The status badges✅ are clickable, and will take you directly to the Action log wherein a summary is generated by :code:`Hypothesis`.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
