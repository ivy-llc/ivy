Setting up
==========

We're really happy you'd like to learn how to contribute towards Ivy 🙂

This page explains the main steps to get started!

Clone Ivy
---------

The first step is simple, clone Ivy!

Depending on your preferred mode of cloning, any of the below should work:

.. code-block:: none

    git clone git@github.com:unifyai/ivy.git

.. code-block:: none

    git clone https://github.com/unifyai/ivy.git

.. code-block:: none

    gh repo clone unifyai/ivy your_folder

PyCharm
-------

`Pycharm <https://www.jetbrains.com/pycharm/>`_ is the main IDE of choice for our development team.
However, you are of course welcome to use whatever Integrated Development Environment (IDE) you're most familiar with.
If you do decide to use PyCharm,
you should make sure to check whether you are eligible for a
`free student licence <https://www.jetbrains.com/community/education/#students>`_.
Many people seem to miss this option,
so we thought we would add an explicit reminder here in the setting up guide!

Docker Interpreter with PyCharm
-------------------------------

Here we show how to set up a specific python environment, which will make contributing much easier.

Setting up and using the same remote python interpreter provided as a docker container helps make sure we are all
using the same packages, and helps to mitigate any potential version conflicts etc.

In addition, it makes it possible to use modules not yet available for a particular operating system,
such as jaxlib on a Windows machine.

Below we provide instructions for setting up a docker interpreter for `Pycharm <https://www.jetbrains.com/pycharm/>`_,
which is the main IDE of choice for our development team:


Windows
****


1. Install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_

2. Install `WSL 2 <https://docs.microsoft.com/en-us/windows/wsl/install>`_. For most, it will only require running the command :code:`wsl --install` in powershell admin mode. Visit the link if it doesn't.

3. Get the latest Docker Image for Ivy by: 
  * Running Docker desktop.
  * Opening cmd, and running the command: :code:`docker pull unifyai/ivy:latest`

4. Install `Pycharm Professional Version <https://www.jetbrains.com/pycharm/>`_
5. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

* Going to settings>project:...>Python Interpreter)

* Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.

* Choosing Docker from the left panel. Type python3 (with the number) in python interpreter path and press ok.

|
DONE. You're set.
|
If Docker's latest version causes error, try using an earlier version by visiting `Docker release note <https://docs.docker.com/desktop/windows/release-notes/>`_


MacOS
****


1. Install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_

2. Get the latest Docker Image for Ivy by: 
  * Running Docker desktop.
  * Opening terminal, and running the command: :code:`docker pull unifyai/ivy:latest`

4. Install `Pycharm Professional Version <https://www.jetbrains.com/pycharm/>`_
5. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

* Going to PyCharm > Preferences > Project:... > Python Interpreter

* Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.

* Choosing Docker from the left panel. Type python3 (with the number) in python interpreter path and press ok.

|
DONE. You're set.
|
If Docker's latest version causes error, try using an earlier version by visiting `Docker release note <https://docs.docker.com/desktop/windows/release-notes/>`_

Ubuntu
****

ToDo: write this section

Pre-Commit
----------

In addition to the docker interpreter,
our development team also make use of the :code:`pre-commit` PyPI `package <https://pypi.org/project/pre-commit/>`_.

Check out their `page <https://pre-commit.com/>`_ for more details.

In a nutshell, this enables us to add pre-commit hooks which check for lint errors before a commit is accepted,
and then also (in most cases) automatically make the necessary fixes.
If the lint tests fail when a commit is attempted, then the commit will not succeed,
and the problematic lines are printed to the terminal. Fixes are then applied automatically where possible.
To proceed with the commit, the modified files must be re-added using git,
and the commit will then succeed on the next attempt.

In order to install and properly set up pre-commit, these steps should be followed:

1. Run :code:`python3 -m pip install pre-commit`

2. Enter enter into your cloned ivy folder, for example :code:`cd ~/ivy`

3. Run :code:`pre-commit install`

That's it! Now when you make a commit, the pre-commit hooks will all be run correctly, as explained above.