Why Unify?
==========

“What is the point of unifying all ML frameworks?” you may ask.

You may be perfectly happy with the framework you currently use, and that’s great! We live in a time where great ML tools are in abundance, and that’s a wonderful thing!

Ivy just makes a wonderful thing **even better**…

We’ll give two clear examples of how Ivy can streamline your ML workflow and save you **weeks** of development time.

No More Re-implementations 🚧
-----------------------------

Let’s say `DeepMind <https://deepmind.com>`_ release an awesome paper in JAX, and you’d love to try it out using your own framework of choice.
Let’s use `PerceiverIO <https://deepmind.com/research/open-source/perceiver-IO>`_ as an example.
What happens currently is:

#. A slew of open-source developers rush to re-implement the code in all ML frameworks, leading to many different versions (`a <https://github.com/lucidrains/perceiver-pytorch>`_, `b <https://github.com/krasserm/perceiver-io>`_, `c <https://github.com/Rishit-dagli/Perceiver>`_, `d <https://github.com/esceptico/perceiver-io>`_, `e <https://github.com/huggingface/transformers/tree/v4.16.1/src/transformers/models/perceiver>`_, `f <https://github.com/keras-team/keras-io/blob/master/examples/vision/perceiver_image_classification.py>`_, `g <https://github.com/deepmind/deepmind-research/tree/21084c8489c34defe7d4e20be89715bba914945c/perceiver>`_).

#. These implementations all inevitably deviate from the original, often leading to: erroneous training, poor convergence, performance issues etc.
   Entirely new papers can even be published for having managed to `get things working in a new framework <https://link.springer.com/chapter/10.1007/978-3-030-01424-7_10>`_.

#. These repositories become full of issues, pull requests, and confusion over why things do or don’t work exactly as expected in the original paper and codebase (`a <https://github.com/lucidrains/perceiver-pytorch/issues>`_, `b <https://github.com/krasserm/perceiver-io/issues>`_, `c <https://github.com/Rishit-dagli/Perceiver/issues>`_, `d <https://github.com/esceptico/perceiver-io/issues>`_, `e <https://github.com/huggingface/transformers/issues>`_, `f <https://github.com/keras-team/keras-io/issues>`_, `g <https://github.com/deepmind/deepmind-research/issues>`_).

#. In total, 100s of hours are spent on: developing each spin-off codebase, testing the code, discussing the errors, and iterating to try and address them.
   This is all for the sake of re-implementing a single project in multiple frameworks.

With Ivy, this process becomes:

#. With one line, convert the code directly to your framework with a computation graph guaranteed to be identical to the original.

We have turned a 4-step process which can take 100s of hours into a 1-step process which takes a few seconds.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/background/why_unify/perceiver_effort.png?raw=true
   :align: center
   :width: 100%

Taking things further, we can use this automatic conversion tool to open up **all** ML tools to **everyone** regardless of their framework.

“Infinite” Shelf-Life ✅
------------------------

Wouldn’t it be nice if we could write some code once and know that it won’t become quickly obsolete among the frantic rush of framework development?

A lot of developers have spent a lot of time porting TensorFlow code to PyTorch in the last few years, with examples being `Lucid <https://github.com/greentfrapp/lucent>`_, `Honk <https://github.com/castorini/honk>`_ and `Improving Language Understanding <https://github.com/huggingface/pytorch-openai-transformer-lm>`_.

The pattern hasn’t changed, developers are now spending many hours porting code to JAX.
For example: `TorchVision <https://github.com/rolandgvc/flaxvision>`_, `TensorFlow Graph Nets library <https://github.com/deepmind/jraph>`_, `TensorFlow Probability <https://github.com/deepmind/distrax>`_, `TensorFlow Sonnet <https://github.com/deepmind/dm-haiku>`_.

What about the next framework that gets released in a few years from now, must we continue re-implementing everything over and over again?

With Ivy, you can write your code **once**, and then it will support all future ML frameworks with **zero** changes needed.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/background/why_unify/future_proof.png?raw=true
   :align: center
   :width: 80%

The same can be said about high-level code for: Modules, Optimizers and Trainers etc.
Currently, the status quo is to continue implementing new high-level libraries for each new framework, with examples being: (a) `Sonnet <https://github.com/deepmind/sonnet>`_, `Keras <https://github.com/keras-team/keras>`_ and `Dopamine <https://github.com/google/dopamine>`_ for TensorFlow (b) `Ignite <https://github.com/pytorch/ignite>`_, `Catalyst <https://github.com/catalyst-team/catalyst>`_, `Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_, and `FastAI <https://github.com/fastai/fastai>`_ for PyTorch, and (c) `Haiku <https://github.com/deepmind/dm-haiku>`_, `Flax <https://github.com/google/flax>`_, `Trax <https://github.com/google/trax>`_ and `Objax <https://github.com/google/objax>`_ for JAX.

With Ivy, we have implemented Modules, Optimizers and Trainers **once** with simultaneous support for all **current** and **future** frameworks.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/background/why_unify/reinvented_wheels.png?raw=true
   :align: center
   :width: 100%

**Round Up**

Hopefully this has given you some idea of the many benefits that a fully unified ML framework could offer 🙂

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!