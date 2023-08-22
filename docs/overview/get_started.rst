Get Started
===========

..

   If you want to use **Ivy's compiler and transpiler**, make sure to follow the 
   `setting up instructions for the API key <https://unify.ai/docs/ivy/compiler/setting_up.html>`_ 
   after installing Ivy!


Depending on your preferred environment you can install Ivy in various ways:

Installing using pip
--------------------

The easiest way to set up Ivy is to install it using pip with the following command:

.. code-block:: bash

    pip install ivy

Keep in mind that this **won't** install any framework other than NumPy!

Docker
------

If you prefer to use containers, we also have pre-built Docker images with all the 
supported frameworks and some relevant packages already installed, which you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest

If you are working on a GPU device, you can pull from:

.. code-block:: bash

    docker pull unifyai/ivy:latest-gpu

Installing from source
----------------------

You can also install Ivy from source if you want to take advantage of the latest 
changes, but we can't ensure everything will work as expected!

.. code-block:: bash

    git clone https://github.com/unifyai/ivy.git
    cd ivy 
    pip install --user -e .


If you are planning to contribute, you want to run the tests, or you are looking 
for more in-depth instructions, it's probably best to check out 
the `Contributing - Setting Up <https://unify.ai/docs/ivy/overview/contributing/setting_up.html#setting-up>`_ page, 
where OS-specific and IDE-specific instructions and video tutorials to install Ivy are available!
