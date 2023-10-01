Building the Docs
=================

This document describes how to build the Ivy docs. If you want to know more about how
our custom building pipeline work, check our `Building the Docs Pipeline
<../deep_dive/building_the_docs_pipeline.rst>`_ deep dive

Building the Docs using Docker
------------------------------

Using convenience script
~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to build the docs is to use the ``docs/make_docs.sh`` script.

.. code-block:: bash

    cd docs
    ./make_docs.sh

This script will build the docs for Ivy and store it in ``docs/build``.

Using existing image on Docker Hub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use the ``unifyai/doc-builder`` image hosted on
`Docker Hub <https://hub.docker.com/r/unifyai/doc-builder>`_ to build the
docs.

Run ``docker run`` to build the docs. The following command will build the docs for
the project in the current directory and output them to ``docs/build``.

.. code-block:: bash

    cd <ivy directory>
    docker run --rm -v $(pwd):/project unifyai/doc-builder

This command will mount the module directory to ``/project`` in the container, the
current directory should be the root of ``ivy``.

Building the image locally
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also build the image locally. You will first need to clone the ``doc-builder``
repository.

Run this command if you are using HTTPS:

.. code-block:: bash

    git clone https://github.com/unifyai/doc-builder.git

Or this command if you are using SSH:

.. code-block:: bash

    git clone git@github.com:unifyai/doc-builder.git

Then, run the following command to build the image:

.. code-block:: bash

    cd doc-builder
    docker build -t unifyai/doc-builder .

Building the Docs without Docker
--------------------------------

You can also build the docs without Docker. You will first need to clone the
``unifyai/doc-builder`` repository. Then use the convenience script
``make_docs_without_docker.sh``.

Run this command if you are using HTTPS:

.. code-block:: bash

    git clone https://github.com/unifyai/doc-builder.git

Or this command if you are using SSH:

.. code-block:: bash

    git clone git@github.com:unifyai/doc-builder.git

Then, run the following command to build the docs:

.. code-block:: bash

    cd doc-builder
    ./make_docs_without_docker.sh <ivy directory>

The script will install the required dependencies for `sphinx <https://www.sphinx-doc.org>`_
which is used to build the docs, as well as dependencies required by Ivy. Then it will
build the docs for Ivy and store it in ``docs/build``.
