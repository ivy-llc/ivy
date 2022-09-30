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
.. _`miniconda`: https://docs.conda.io/en/latest/miniconda.html
.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`ivy/run_tests_CLI`: https://github.com/unifyai/ivy/tree/f71a414417646e1dfecb5de27fb555f80333932c/run_tests_CLI

We're really happy you'd like to learn how to contribute towards Ivy 🙂

This page explains the main steps to get started!

Clone Ivy
---------

The first step is simple, clone Ivy!

Depending on your preferred mode of cloning, any of the below should work:

.. code-block:: none

    git clone --recurse-submodules git@github.com:unifyai/ivy.git

.. code-block:: none

    git clone --recurse-submodules https://github.com/unifyai/ivy.git

.. code-block:: none

    gh repo clone --recurse-submodules unifyai/ivy your_folder

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

Virtual environments - No Docker
-------------------------------

Due to the rapid pace of updates in Ivy, it is strongly suggested for developers to use the latest
ivy package from GitHub source, as explained below. This is to ensure the contributors' code and
examples are as aligned and in accordance with the latest as possible. The stable version of Ivy
from PyPI maybe used for personal projects and experiments but avoided in development, for now. If you
want to use the stable version, you are welcome to use the docker container or pip install ivy-core.

Below is a guide to creating your own virtual environment. The benefit of creating a python environment
is the ability to install certain packages for a project and then other packages (perhaps different versions) in a
new environment for another project. This makes it very easy to keep track of installed packages and their versions.

Below is a guide for setting up a developing environment for Ivy.

You can either use `miniconda`_ or `venv`_:

Using miniconda
****

#. Install `miniconda`_
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

#. Setup the interpreter from you environment in Pycharm by:

   a. Going to settings -> project -> Python Interpreter

   b. Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.

   c. Choosing "conda environment" from the left panel. Choose existing environment and select the drop down and you should find the path python in the environment.
If you don't find path to you created python environment, you can run :code:`where python` in conda command line while the environment is activate and it should give the path which can be added manually.


Using venv
****
This is a builtin package and doesn't require explicit installation.

#. Open your terminal/cmd in the directory where you would like to have the folder with the environment files
#. Create the environment by running the command below with a new environment name. We named it :code:`ivy_dev` like above.

    .. code-block:: none

        python -m venv ivy_dev

    Try :code:`python3` if :code:`python` doesn't work.

#. Activate the created environment by running (in the same working directory as the environment folder):

    .. code-block:: none

        ivy_dev\Scripts\activate.bat

    (on Windows)

    OR

    .. code-block:: none

        source ivy_dev/bin/activate

    (on Mac/Linux)

#. Now install ivy package from GitHub by running:

    .. code-block:: none

        pip install git+https://github.com/unifyai/ivy.git

#. Setup the interpreter from you environment in Pycharm by:

   a. Going to settings -> project -> Python Interpreter

   b. Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.

   c. Choosing "virtualenv environment" from the left panel. Choose existing environment and add the path to python.
The path to python can be found by :code:`where python` on Windows and :code:`which python` in Linux/Mac OS.

Note: You may tick "Make available to all projects" so you will be able to find the interpreter from the conda/venv environment in any
future projects.

To make sure you have all the packages for running tests available change the directory to :code:`ivy/ivy_tests/test_array_api` in your
cloned fork using the :code:`cd` command and run the command below (while your :code:`ivy_dev` environment is active):

    .. code-block:: none

        pip install -r requirements.txt

This will install packages required for running the tests in Array API suite.

Here are the visual guides for setting up a `virtualenv environment <https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#0>`_
OR `conda environment <https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html>`_ in pycharm from JetBrains.

Docker Interpreter with PyCharm
-------------------------------


Setting up and using the same remote python interpreter provided as a docker container helps make sure we are all
using the same packages (same environment) and helps to mitigate any potential version conflicts etc.

In addition, it makes it possible to use modules not yet available for a particular operating system,
such as :code:`jaxlib` on a Windows machine.

Below, we provide instructions for setting up a docker interpreter for `Pycharm <https://www.jetbrains.com/pycharm/>`_,
which, as mentioned above, is the IDE of choice for our development team:


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


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/5NKOLrn-9BM" class="video">
    </iframe>


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

Setting Up Testing
******************
There are a couple of options to choose from when running ivy tests in PyCharm. To run a single unit test, e.g. `test_abs`,
you can avail of the context menu in the PyCharm code editor by pressing the green ▶️ symbol which appears to the left
of `def test_abs(`.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/setting_up_testing/pycharm_test_run_1.png?raw=true
  :width: 420

You can then click 'Run pytest for...' or 'Debug pytest for...'. Keyboard shortcuts for running the rest are displayed
also. These screenshots are from a Mac, hence the shortcut for running a test is :code:`ctrl - shift - R`.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/setting_up_testing/pycharm_test_run_2.png?raw=true
  :width: 420

The test run should pop up in a window at the bottom of the screen (or elsewhere, depending on your settings).

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/setting_up_testing/pycharm_test_run_3.png?raw=true
  :width: 420

To run all the tests in a file, press :code:`ctrl` - right click (on Mac) on the :code:`test_elementwise.py` open tab.
A menu will appear in which you can find 'Run pytest in test_elementwise.py...'

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/setting_up_testing/pycharm_run_all_1.png?raw=true
  :width: 420

Click this and you should see a progress bar of all the tests running in the file.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/setting_up_testing/pycharm_run_all_2.png?raw=true
  :width: 420

It is also possible to run the entire set of ivy tests or the array api test suite using pre-written shell scripts that
can be run from the 'Terminal' tab in PyCharm. There are a number of such shell scripts in `ivy/run_tests_CLI`_:

.. code-block:: bash
    :emphasize-lines: 4,5,8,9,10

    run_ivy_core_test.py
    run_ivy_nn_test.py
    run_ivy_stateful_test.py
    run_tests.sh
    test_array_api.sh
    test_dependencies.py
    test_dependencies.sh
    test_ivy_core.sh
    test_ivy_nn.sh
    test_ivy_stateful.sh

* :code:`run_tests.sh` is run by typing :code:`./run_tests_CLI/run_tests.sh` in the :code:`/ivy` directory. This runs all tests in :code:`ivy/ivy_tests`.
* :code:`test_array_api.sh` is run by typing :code:`./test_array_api.sh [backend] test_[submodule]`. This runs all array-api tests for a certain submodule in a certain backend.
* :code:`test_ivy_core.sh` is run by typing :code:`./run_tests_CLI/test_ivy_core.sh [backend] test_[submodule]` in the ivy directory. This runs all ivy tests for a certain submodule in a certain backend in :code:`test_ivy/test_functional/test_core`.
* :code:`test_ivy_nn.sh`, :code:`test_ivy_stateful.sh` are run in a similar manner to :code:`test_ivy_core.sh`. Make sure to check the submodule names in the source code before running.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/setting_up_testing/pycharm_run_array_api_tests.png?raw=true
  :width: 420

If you wish to run tests of all submodules of `ivy_core`, `ivy_nn` or `ivy_stateful`, there are :code:`.py` available
in :code:`run_tests_CLI`. All are run like:
:code:`python run_tests_CLI/run_ivy_nn_test.py 1`, where 1 = numpy, 2 = torch, 3 = jax, and 4 = tensorflow.


More Detailed Hypothesis Logs
****
For testing, we use the `Hypothesis <https://hypothesis.readthedocs.io/en/latest/#>`_ module for data generation.
During testing, if Hypothesis detects an error, it will do its best to find the simplest values that are causing the error.
However, when using PyCharm, if Hypothesis detects two or more distinct errors, it will return the number of errors found and not return much more information.
This is because PyCharm by default turns off headers and summary's while running tests. To get more detailed information on errors in the code, we recommend doing the following:

#. Going to the settings -> Advanced
#. Using the search bar to search for 'Pytest'
#. Make sure that the checkbox for 'Pytest: do not add "--no-header --no-summary -q"' is checked.

    a. .. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/more_detailed_hypothesis_logs/detailed_hypothesis_setting.png?raw=true
          :width: 420

Now, if Hypothesis detects an error in the code it will return more detailed information on each of the failing examples:

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/more_detailed_hypothesis_logs/detailed_hypothesis_example.png?raw=true
   :width: 420

GitHub Codespaces
-----------------

It can be headache to install Docker and setup the PyCharm development environment, especially on recent ARM architectures like the new M1 Macs. Instead, we could make use of the GitHub Codespaces feature provided; this feature creates a VM (Virtual Machine) on the Azure cloud (means no local computation) with same configuration as defined by :code:`ivy/Dockerfile`. Since it's a VM, we no longer have to worry about installing the right packages, modules etc., making it platform agnostic (just like ivy :P). We can develop as we usually do on Visual Studio Code with all your favourite extensions and themes available in Codespaces too. With all the computations being done on cloud, we could contribute to Ivy using unsupported hardware, old/slow systems, even from your iPad as long as you have visual studio code or a browser installed. How cool is that ?!

**Pre-requisites**

1. Before we setup GitHub Codespaces, we need to have Visual Studio Code installed (you can get it from `here <https://code.visualstudio.com/>`_). 

2. Once Visual Studio Code is installed, head over to the extension page (it's icon is on the left pane), and search "Codespaces" and then install the extension locally.

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/extension_install.png?raw=true
   :width: 420

Now we are ready to begin!

**Setting up Codespaces**

Just follow the steps outlined below:

1. Go to your fork of :code:`ivy`, and then click on the green "Code" dropdown, go to Codespaces tab, and then click on "create codespace on master".

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/fork_create_codespace.png?raw=true
   :width: 420

2. This will open up a new tab, where you click on "Open this codespaces on VS code desktop". Give the relevant permissions to the browser to open up Visual Studio Code.

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/open_vscode_desktop.png?raw=true
   :width: 420

3. Once visual studio code opens up, it will start building the remote container. In order to view the logs while the container is being built, you may click on "Building Codespace..." on the bottom right box. Please be patient while container is being built, it may take upto 10-15 minutes, but it's a one-time process. Any subsequent connections to your ivy codespace will launch in 10-12 seconds.

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/building_codespace.png?raw=true
   :width: 420

Log of container being built would look like below:

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/log_codespace.png?raw=true
   :width: 420

4. Once the container is built, you would see the following output log saying "Finished configuring codespace".

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/codespace_built.png?raw=true
   :width: 420

5. That's it, you have just setup GitHub codespaces and can start developing Ivy. The configuration files installs all the required packages, extensions for you to get started quickly.

**Opening an existing Codespace**

If you have already setup codespaces, refer to the following to open your previously setup codespaces environment.

There are 3 ways to connect your existing codespaces, you can use any of the approaches mentioned below.

1. Go to your fork of ivy, click on the green coloured dropdown "Code", go to codespaces tab, then select your codespace. This will open up a new tab, from there either you can develop on the browser itself, or click on "Open this codespaces on VS code desktop" to open up visual studio code application and develop from there.

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/existing_codespace_fork.png?raw=true
   :width: 420

2. Other way to connect is to open up visual studio code application. There is a good chance that you would see :code:`ivy [Codespaces]` or :code:`ivy [vscode-remote]` on your recently opened projects. If you click either of those, it will open up your codespace. 

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/recent_projects.png?raw=true
   :width: 420

3. If in any case it doesn't show your codespace on recent projects, go to "Remote Connection Explorer" extension tab on the left pane, from there make sure you have selected "Github Codespaces" on the top-left dropdown. Once you find your codespace, right click on it and then select "Connect to codespace in current window".

.. image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/master/img/externally_linked/contributing/0_setting_up/github_codespaces/connect_existing.png?raw=true
   :width: 420

**Troubleshooting**

Sometimes, visual studio code is not able to select the python interpreter. However, you can do that manually if that ever happens. Open up any python file, then click on the bottom right where it is written "Select Python Interpreter". From there, select :code:`Python 3.8.10 64-bit usr/bin/python3`.

**Setting Up Testing**

The steps are as following to setup testing on VS Code when using a new Codespace.

1. Under the flask Icon in the toolbar select "Configure Python Tests" and select PyTest as the test framework.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/vs_code_testing_setup/vs_testing_01.png?raw=true
   :width: 420

2. Select ivy_tests as the root directory for testing.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/vs_code_testing_setup/vs_testing_02.png?raw=true
   :width: 420

3. Configure the _array_module.py file in the array_api_tests to be set to one of the supported frameworks.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/vs_code_testing_setup/vs_testing_03.png?raw=true
   :width: 420

4. As of 01/08/2022, the conftest.py file in the array_api_tests folder must also be commented out in order to run ivy_tests in the test suite. This will cause the array_api_tests to fail and therefore they must be run via the terminal.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/contributing/0_setting_up/vs_code_testing_setup/vs_testing_04.png?raw=true
   :width: 420

5. Following all of this you should refresh the test suite and you should now be able to run tests right from VS Code!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/8rDcMMIl8dM" class="video">
    </iframe>


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
