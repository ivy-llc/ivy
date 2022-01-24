Using Ivy
=========

Ivy is useful tool for any developer wishing to maximize the audience of their codebase,
by making their code framework-agnostic.

First, we explain how to write framework-agnostic Ivy code.
We then explain the variety of ways in which an end user can run your Ivy code,
using their specific choice of backend framework.

Writing Ivy
-----------

Writing Ivy code is simple.
The :code:`ivy` namespace provides access to all of the Ivy functions,
which can all be used directly with tensors from any of the supported machine learning frameworks.

Let's assume the codebase that you wish to share with the community consists of a single simple function, outlined below.
In reality, your code will likely be a little more complex than this!


.. code-block:: python

    import ivy

    def sincos(x):
        return ivy.sin(ivy.cos(x))


That's it, having used the Ivy namespace to write your code, it is now framework-agnostic! There is nothing more to it!

Running Ivy
-----------

When it comes to an end user actually *running* your code alongside their specific framework, there are a number of options.

**Type Checking**

Firstly, the user can simply use your new function directly in their code, like so:

.. code-block:: python

    import tensorflow as tf
    from your_awesome_project import sincos

    # some exciting tf code
    x = tf.constant([1.])
    y = sincos(x)  # with type checking
    # more exciting tf code

In this simple format, each :code:`ivy` method will simply type-check the inputs to infer the correct backend framework,
and then call the framework-specific functions. A PyTorch user's code would look like this:

.. code-block:: python

    import torch
    from your_awesome_project import sincos

    # some exciting torch code
    x = torch.tensor([1.])
    y = sincos(x)  # with type checking
    # more exciting torch code

This is the simplest way for an end user to run your framework-agnostic code.
However, in eager execution mode the continual type-checking adds a small overhead,
which we may wish to avoid.

**Framework Setting**

To avoid the small type-checking overhead, the framework can be set explicitly, by calling :code:`ivy.set_framework("torch")` for example.
This updates the :code:`__dict__` of the :code:`ivy` namespace with the :code:`__dict__` of a framework-specific namespace,
such as :code:`ivy.backends.torch`. The framework-specific namespace bypasses type-checking of the inputs.
The end user would use your codebase like so:

.. code-block:: python

    import ivy
    import torch
    from your_awesome_project import sincos

    ivy.set_framework('torch')
    # some exciting torch code
    x = torch.tensor([0.])
    y = sincos(x)  # no type checking
    # more exciting torch code
    ivy.unset_framework()

**Framework Block Setting**

Finally, some users may wish to use multiple frameworks in a single project.
For example, NumPy is often used in conjunction with DL frameworks.
NumPy can be used for implementing non-differentiable parts of the project on the CPU,
and interfacing with other packages such as OpenCV and MatPlotLib for visualization.

For end users wishing to use different parts of their code with different backends in a single project,
without using repeated calls to :code:`ivy.set_framework()` and :code:`ivy.unset_framework()`,
then the :code:`with` statement can be used like so:

.. code-block:: python

    import ivy
    import torch
    import numpy as np
    from your_awesome_project import sincos

    with ivy.backends.numpy.use:
        # some exciting np code
        x = np.array([0.])
        y = sincos(x)  # no type checking
        # more exciting np code

    with ivy.backends.torch.use:
        # some exciting torch code
        x = torch.tensor([0.])
        y = sincos(x)  # no type checking
        # more exciting torch code

**User Flexibility**

Overall, this variety in backend selection gives end users a lot of flexibility.
If they want to dive straight into using your Ivy project, then they can start using functions immediately,
with type-checking in the background. If they use only a single backend in their project,
they can explicitly set this at the beginning of their own code using :code:`ivy.set_framework()`.
Finally, if they use multiple backends in one project,
they can set and unset the frameworks using either :code:`ivy.set_framework()` and :code:`ivy.unset_framework()`,
or via the :code:`with` statement.