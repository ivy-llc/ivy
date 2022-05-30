Setting up
==========
This page provides the instructions to set up a specific python environment to contribute towards Ivy.
Setting up and using the same remote python interpreter provided as a docker container helps the community to share the
same packages and mitigate any version conflicts.
In addition, it makes it possible to use modules not yet provided for an operating system, such as jaxlib for windows.
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
  * Opening terminal, and running the command: :code:`docker run -it --rm unifyai/ivy:latest python3`

4. Install `Pycharm Professional Version <https://www.jetbrains.com/pycharm/>`_
5. Open pycharm with your cloned Ivy repository. Add the remote python interpreter by:

* Going to PyCharm > Preferences > Project:... > Python Interpreter

* Clicking add interpreter (currently by clicking the ⚙ icon by the right side) which should open a new window.

* Choosing Docker from the left panel. Type python3 (with the number) in python interpreter path and press ok.

|
DONE. You're set.
|
If Docker's latest version causes error, try using an earlier version by visiting `Docker release note <https://docs.docker.com/desktop/windows/release-notes/>`_
