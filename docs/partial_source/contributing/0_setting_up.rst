Setting Up
==========

.. _`setting up discussion`: https://github.com/unifyai/ivy/discussions/1308
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`pycharm channel`: https://discord.com/channels/799879767196958751/942114831039856730
.. _`docker channel`: https://discord.com/channels/799879767196958751/942114744691740772
.. _`pre-commit channel`: https://discord.com/channels/799879767196958751/982725464110034944
.. _`pip packages channel`: https://discord.com/channels/799879767196958751/942114789642080317
.. _`other channel`: https://discord.com/channels/799879767196958751/982727719836069928

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

For questions, please reach out on the `setting up discussion`_
or on `discord`_ in the `pycharm channel`_!

Due to rapid pace of updates in Ivy, it is strongly suggested for developers to use the latest
ivy package using GitHub, as explained below. This is to ensure the contributors code and
examples are as aligned and in accordance with the latest as possible. The stable version of Ivy
from PyPI maybe used for personal projects and experiments but avoided in development, for now. If you
want to experiment with stable version, you can use docker.


Virtual environments - No Docker
-------------------------------

Using miniconda
****

#. Install miniconda
#. Open conda terminal
#. Create the environment by running the command (:code:`ivy_dev` is the name of the environment)

    .. code-block:: none

        conda create --name ivy_dev python=3.8.10

#. Activate the environment by:

    .. code-block:: none

        conda activate ivy_dev

#.  Now install ivy package from GitHub by running:

    .. code-block:: none

        pip install git+https://github.com/unifyai/ivy.git



Once you have created yoiu

Using venv
****


Docker Interpreter with PyCharm
-------------------------------

Here we show how to set up a specific python environment, which will make contributing much easier.

Setting up and using the same remote python interpreter provided as a docker container helps make sure we are all
using the same packages, and helps to mitigate any potential version conflicts etc.

In addition, it makes it possible to use modules not yet available for a particular operating system,
such as :code:`jaxlib` on a Windows machine.

Below we provide instructions for setting up a docker interpreter for `Pycharm <https://www.jetbrains.com/pycharm/>`_,
which, as mentioned above, is the main IDE of choice for our development team:


Windows
****


#. Install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_
#. Install `WSL 2 <https://docs.microsoft.com/en-us/windows/wsl/install>`_. For most, it will only require running the command :code:`wsl --install` in powershell admin mode. Visit the link if it doesn't.
#. Get the latest Docker Image for Ivy by:

   a. Running Docker desktop.
   b. Opening cmd, and running the command: :code:`docker pull unifyai/ivy:latest`

#. Install `Pycharm Professional Version <https://www.jetbrains.com/pycharm/>`_
#. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

   a. Going to the settings -> Build, Execution, Deployment -> Docker. Click the "+" on top left and it should add a docker connection.
   b. Going to settings -> project -> Python Interpreter
   c. Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.
   d. Choosing "Docker" from the left panel. Type python3 (with the number) in python interpreter path and press ok.

Once these steps are finished, your interpreter should be set up correctly!
If Docker's latest version causes error,
try using an earlier version by visiting
`Docker release note <https://docs.docker.com/desktop/release-notes/>`_.
For some Windows users, it might be necessary to enable virtualisation from the BIOS setup.

MacOS
****


#. Install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_
#. Get the latest Docker Image for Ivy by:

   a. Running Docker desktop.
   b. Opening terminal, and running the command: :code:`docker pull unifyai/ivy:latest`

#. Install `Pycharm Professional Version <https://www.jetbrains.com/pycharm/>`_
#. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

   a. Going to the settings -> Build, Execution, Deployment -> Docker. Click the "+" on top left and it should add a docker connection.
   b. Going to settings -> project -> Python Interpreter
   c. Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.
   d. Choosing "Docker" from the left panel. Type python3 (with the number) in python interpreter path and press ok.

Once these steps are finished, your interpreter should be set up correctly!
If Docker's latest version causes error,
try using an earlier version by visiting
`Docker release note <https://docs.docker.com/desktop/release-notes/>`_.

Ubuntu
****


#. Install Docker by running the commands below one by one in the Linux terminal. You may
   visit `Docker Ubuntu Installation Page <https://docs.docker.com/engine/install/ubuntu/>`_ for the details.

    .. code-block:: none

        sudo apt-get update

    .. code-block:: none

        sudo apt-get install \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    .. code-block:: none

        sudo mkdir -p /etc/apt/keyrings

    .. code-block:: none

        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    .. code-block:: none

        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    .. code-block:: none

        sudo apt-get update

    .. code-block:: none

        sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

#. Get the latest Docker Image for Ivy by:

   a. Opening terminal and running :code:`systemctl start docker`
   b. Running the command: :code:`docker pull unifyai/ivy:latest`

   Note: If you get permission related errors please visit the simple steps at `Linux post-installation page. <https://docs.docker.com/engine/install/linux-postinstall/>`_

#. Install Pycharm Professional Version. You may use Ubuntu Software for this.
#. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

   a. Going to the settings -> Build, Execution, Deployment -> Docker. Click the "+" on top left and it should add a docker connection.
   b. Going to settings -> project -> Python Interpreter
   c. Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.
   d. Choosing "Docker" from the left panel. Type python3 (with the number) in python interpreter path and press ok.

For questions, please reach out on the `setting up discussion`_
or on `discord`_ in the `docker channel`_!

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

2. Enter into your cloned ivy folder, for example :code:`cd ~/ivy`

3. Run :code:`pre-commit install`

That's it! Now when you make a commit, the pre-commit hooks will all be run correctly,
as explained above.

For questions, please reach out on the `setting up discussion`_
or on `discord`_ in the `pre-commit channel`_!

**Round Up**

This should have hopefully given you a good understanding of how to get things properly set up.

If you're ever unsure of how best to proceed,
please feel free to engage with the `setting up discussion`_,
or reach out on `discord`_ in the `pycharm channel`_, `docker channel`_,
`pre-commit channel`_, `pip packages channel`_ or `other channel`_,
depending on the question!