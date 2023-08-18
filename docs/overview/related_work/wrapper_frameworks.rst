.. _`RWorks Wrapper Frameworks`:

Wrapper Frameworks
==================

.. _`EagerPy`: https://eagerpy.jonasrauber.de/
.. _`PyTorch`: https://pytorch.org/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`JAX`: https://jax.readthedocs.io/
.. _`NumPy`: https://numpy.org/
.. _`Keras`: https://keras.io/
.. _`Microsoft Cognitive Toolkit`: https://learn.microsoft.com/en-us/cognitive-toolkit/
.. _`Theano`: https://github.com/Theano/Theano
.. _`PlaidML`: https://github.com/plaidml/plaidml
.. _`Thinc`: https://thinc.ai/
.. _`MXNet`: https://mxnet.apache.org/
.. _`TensorLy`: http://tensorly.org/
.. _`NeuroPod`: https://neuropod.ai/
.. _`CuPy`: https://cupy.dev/
.. _`SciPy`: https://scipy.org/
.. _`TorchScript`: https://pytorch.org/docs/stable/jit.html
.. _`discord`: https://discord.gg/sXyFF8tDtm

.. |eagerpy| image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/related_work/wrapper_frameworks/eagerpy.png
    :height: 15pt
    :class: dark-light
.. |keras| image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/related_work/wrapper_frameworks/keras.png
    :height: 20pt
    :class: dark-light
.. |thinc| image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/related_work/wrapper_frameworks/thinc.png
    :height: 15pt
.. |tensorly| image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/related_work/wrapper_frameworks/tensorly.png
    :height: 20pt

There are a variety of wrapper frameworks which wrap around other ML frameworks, enabling these ML frameworks to be switched in and out very easily in the backend, and enabling framework-agnostic code to be written, both for deployment and for training.
These wrapper frameworks can be considered as “higher level” than the individual ML frameworks that they wrap, given that they abstract these ML frameworks into the backend, and they typically do not go any lower level than this, often being pure Python projects, delegating all lower level compiler code handling to the frameworks being wrapped.

EagerPy |eagerpy|
-----------------
`EagerPy`_ lets users write code which automatically works natively with `PyTorch`_, `TensorFlow`_, `JAX`_, and `NumPy`_.
Key differences to Ivy are the lack of transpiler support and the lack of a stateful API for constructing high level classes such as network layers, optimizers, initializers and trainers in the framework.

Keras |keras|
-------------
`Keras`_ includes high level classes for building network layers, optimizers, initializers and trainers, and also a lower level functional API.
Up until version 2.3, Keras supported multiple backends, including `TensorFlow`_, `Microsoft Cognitive Toolkit`_, `Theano`_, and `PlaidML`_, but as of version 2.4, only TensorFlow is supported.

Thinc |thinc|
-------------
`Thinc`_ is a lightweight library which offers a functional-programming API for composing models, with support for layers defined in `PyTorch`_, `TensorFlow`_ or `MXNet`_.
Thinc can be used as an interface layer, a standalone toolkit or a way to develop new models.
The focus is very much on high level training workflows, and unlike `EagerPy`_ and `Keras`_, the framework does not implement an extensive functional API at the array processing level.
For example, common functions such as :func:`linspace`, :func:`arange`, :func:`scatter`, :func:`gather`, :func:`split`, :func:`unstack` and many more are not present in the framework.
Thinc instead focuses on tools to compose neural networks based on the most common building blocks, with high level APIs for: Models, Layers, Optimizers, Initializers, Schedules and Losses.

TensorLy |tensorly|
-------------------
`TensorLy`_ provides utilities to use a variety of tensor methods, from core tensor operations and tensor algebra to tensor decomposition and regression.
It supports `PyTorch`_, `Numpy`_, `CuPy`_, `JAX`_, `TensorFlow`_, `MXNet`_ and `SciPy`_ in the backend.
The API is fully functional and strongly focused on high dimensional tensor methods, such as :code:`partial_SVD`, :code:`kron`  and :code:`tucker_mode_dot`, and it does not include a stateful API for constructing high level classes such as network layers, optimizers, initializers and trainers.
There is also no support for some simpler and more common array processing functions such as :func:`scatter`, :func:`gather`, :func:`minimum`, :func:`maximum`, :func:`logical_or`, :func:`logical_and` and many others.

NeuroPod
--------
`Neuropod`_ is a library that provides a uniform interface to run deep learning models from multiple frameworks in C++ and Python.
Neuropod makes it easy for researchers to build models in a framework of their choice while also simplifying deployment of these models.
It currently supports `TensorFlow`_, `PyTorch`_, `TorchScript`_, and `Keras`_.
Compared to other wrapper frameworks, NeuroPod is very high level.
It wraps entire models which have already been trained, in a manner where the interface to these models is unified.
It excels in a setting where multiple networks, which may have been trained in a variety of frameworks, must all act as subsystems performing specific tasks as part of a larger complex system, and the network interfaces in this larger system should be unified.
This abstraction enables subsystem networks to be quickly replaced by other networks performing the same role, irrespective of which framework the subsystem is running under the hood.
