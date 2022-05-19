Setting up
==========
This page provides the instructions to set up a specific python environment to contribute towards Ivy. Setting up and using the same remote python interpreter provided as a docker container helps the community to share the same packages and mitigate any version conflicts. In addition, it makes it possible to use modules not yet provided for an operating system, such as jaxlib for windows.
Below are the instructions required to setup the docker container for:


Windows
****


1. Install `Docker Desktop <https://www.docker.com/products/docker-desktop>`_

2. Install `WSL 2 <https://docs.microsoft.com/en-us/windows/wsl/install>`_. For most, it will only require running the command :code:`wsl --install` in powershell admin mode. Visit the link if it doesn't.

3. Get the latest Docker Image for Ivy by: 
  * Running Docker desktop.
  * Opening cmd, and running the command: :code:`docker run -it --rm unifyai/ivy:latest python3`

4. Install `Pycharm Professional Version <https://www.jetbrains.com/pycharm/>`_
5. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

* Going to settings>project:...>Python Interpreter)

* Choose add interpreter (currently by clicking the setting logo on the side) (Fig. 1 Below)

* Choose Docker from left panel, make sure it says python3 (with the number) in python interpreter path and click ok. (Fig 2 Below)
|
.. figure:: https://user-images.githubusercontent.com/53497039/156894436-d09fcddf-aff1-4514-9536-50f77badff4e.png
    :width: 520px
    :align: left
    :height: 164px
    :alt: alternate text
    :figclass: align-left

    Fig. 1.
|
|
|
|
|
|
|
.. figure:: https://user-images.githubusercontent.com/53497039/156894484-d02f055d-d7eb-4588-8f8e-8d7ffc60bec5.png
    :width: 520px
    :align: left
    :height: 387px
    :alt: alternate text
    :figclass: align-left

    Fig. 2.
|
DONE. You're set.
|
If Docker's latest version causes error, try using an earlier version by visiting `Docker release note <https://docs.docker.com/desktop/windows/release-notes/>`_
