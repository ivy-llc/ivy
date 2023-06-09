Applied Libraries
=================

.. _`Ivy Robot`: https://unify.ai/docs/robot/
.. _`Mech`: https://unify.ai/docs/mech/
.. _`Vision`: https://unify.ai/docs/vision/
.. _`Demo Utils`: https://github.com/unifyai/demo-utils
.. _`Ivy`: https://github.com/unifyai/ivy
.. _`Docker Desktop`: https://www.docker.com/products/docker-desktop/
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`pycharm channel`: https://discord.com/channels/799879767196958751/942114831039856730
.. _`docker channel`: https://discord.com/channels/799879767196958751/942114744691740772
.. _`pre-commit channel`: https://discord.com/channels/799879767196958751/982725464110034944
.. _`pip packages channel`: https://discord.com/channels/799879767196958751/942114789642080317

Introduction
------------

Helping to contribute towards the ivy libraries requires a slightly more complex setup than is needed for contributing to ivy alone.
For instance, `Ivy Robot`_ depends on `Mech`_, `Vision`_ and `Demo Utils`_.
Thus, the related repositories have to be pulled into the same local folder, and `Ivy`_ must also be pulled into this same folder.

To have a better grasp, let's look at an example for Ivy Robot in the next section!

Example - Ivy Robot
-------------------

**Directory Tree**

1. Due to dependencies, the related Ivy repositories have to be placed in the same local directory:

.. code-block:: none

    |-- your-local-dir
    |   |-- ivy
    |   |-- mech
    |   |-- vision
    |   |-- robot
    |   |-- demo-utils

2. Clone all repositories into a mutual directory:

    .. code-block:: none

        git clone https://github.com/unifyai/ivy.git

    .. code-block:: none

        git clone https://github.com/unifyai/mech.git

    .. code-block:: none

        git clone https://github.com/unifyai/vision.git

    .. code-block:: none

        git clone https://github.com/unifyai/robot.git

    .. code-block:: none

        git clone https://github.com/unifyai/demo-utils.git

3. The next steps will depend on your type of development.

**Local Development**

1. Create a virtual environment (venv) in the same directory:

    .. code-block:: none

        python3 -m venv ivy_dev

2. Activate the environment:

    (on Windows)
        .. code-block:: none

            ivy_dev\Scripts\activate.bat

    (on Mac/Linux)
        .. code-block:: none

            source ivy_dev/bin/activate

3. Go into each directory and install packages in develop/editable mode:

    .. code-block:: none

        cd ivy
        python3 -m pip install --user -e .

    (repeat for all repositories)

    **NOTE:** In develop mode, packages are linked to their local directory.
    Therefore, changes or edits are reflected immediately when in use.

4. To use:

    .. code-block:: none

        python3

    .. code-block:: python

        import ivy_robot

**Docker Development**

1. Install `Docker Desktop`_

2. Go into the :code:`robot` repository and build the docker image:

    .. code-block:: none

        cd robot
        docker build -t my-robot .

3. To use, first mount the local directories, then start up :code:`python3` with Docker:

    (in the folder containing all repositories)
        .. code-block:: none

            docker run --rm -it -v `pwd`/ivy:/ivy -v `pwd`/mech:/mech -v `pwd`/vision:/vision -v `pwd`/robot:/robot -v `pwd`/demo-utils:/demo-utils my-robot python3

    **NOTE:** Mounting allows the docker container to use local folder as volumes, thus reflecting the local changes or edits made.
    Users are not required to rebuild the docker image after every change.

**IDE Development**

1. For **PyCharm**, configurations are saved in the :code:`.idea` folder (part of the ivy repo).

2. For **VSCode**, configurations can be found in the :code:`.devcontainer` folder (not part of the ivy repo).

**NOTE:** To use development container in VSCode, the extension "Remote - Containers" needs to be installed.

**NOTE:** When using GitHub Codespaces, the :code:`mounts` config in :code:`.devcontainer/devcontainer.json` is not supported.

**Round Up**

These examples should hopefully give you a good understanding of what is required when developing the Ivy applied libraries.

If you have any questions, please feel free to reach out on `discord`_ in the `pycharm channel`_, `docker channel`_, `pre-commit channel`_, `pip packages channel`_ or `other channel`_, depending on the question!
