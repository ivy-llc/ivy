Ivy Applied Libraries Development Guide
=======================================

.. _`Ivy Robot`: https://lets-unify.ai/robot/
.. _`Mech`: https://lets-unify.ai/mech/
.. _`Vision`: https://lets-unify.ai/vision/
.. _`Demo Utils`: https://github.com/unifyai/demo-utils
.. _`Docker Desktop`: https://www.docker.com/products/docker-desktop/

Introduction
------------

Helping to contribute towards the ivy libraries requires a slightly more complex
setup than it is needed for contributing to ivy alone. For instance, `Ivy Robot`_ is
depending on `Mech`_, `Vision`_ and `Demo Utils`_. Thus, the related repositories
have to be pulled into the same local folder, and ???

To have a better grasp, let's look at an example on Ivy Robot in the next section!

Example - Ivy Robot
-------------------

**Directory Tree**

1. Due to dependencies, the related Ivy repositories have to be placed in the same local folder:

.. code-block:: none

    |-- your-local-dir
    |   |-- ivy
    |   |-- mech
    |   |-- vision
    |   |-- robot
    |   |-- demo-utils

2. Clone all repositories into a mutual folder:

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

1. Create a virtual environment (venv) in the same folder:

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

    **NOTE:** In develop mode, packages are linked to their local directory. Therefore,
    changes or edits are allowed and reflected immediately.

4. To use:

    .. code-block:: none

        python3

    .. code-block:: python

        import ivy_robot


**Docker Development**

1. Install `Docker Desktop`_

2. Go into the ::code::`robot` repository and build the docker image:

    .. code-block:: none

        cd robot
        docker build -t my-robot .

3. To use, first mount the local directories, then start up python3 with Docker:

    (in the folder containing all repositories)
    .. code-block:: none

        docker run --rm -it -v `pwd`/ivy:/ivy -v `pwd`/mech:/mech -v `pwd`/vision:/vision -v `pwd`/robot:/robot -v `pwd`/demo-utils:/demo-utils my-robot python3

    **NOTE:** Mounting allows the docker container to use local folder as volumes, thus
    reflecting the local changes or edits made. Users are not required to rebuild
    the docker image after every change.

**IDE Development**

1. For **PyCharm**, configurations are saved in the `.idea` folder.

2. For **VSCode**, configurations can be found in the `.vscode` folder.

**Round Up**

These examples should hopefully give you a good understanding of what is required
when developing on the Ivy applied libraries.

???
If you're ever unsure of how best to proceed,
please feel free to engage with the `docstrings discussion`_,
or reach out on `discord`_ in the `docstrings channel`_!
