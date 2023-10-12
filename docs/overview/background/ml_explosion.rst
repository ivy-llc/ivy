ML Explosion
============

The number of open source ML projects has grown considerably in recent years, especially Deep Learning, as can be seen from the rapidly increasing number of GitHub repos containing the term “Deep Learning” over time.
These projects are written in many different frameworks.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/background/ml_explosion/num_dl_repos_over_time.png?raw=true
   :align: center
   :width: 80%

While this is a wonderful thing for researchers and developers, when we also consider the speed at which the frameworks are evolving, the shareability of code is significantly hindered, with projects and libraries becoming outdated in a matter of months if not rigorously maintained against the newest frameworks and also the newest framework versions.

For software development pipelines where rapid prototyping and collaboration are vital, this is a significant bottleneck.
As new future frameworks become available, backend-specific code quickly becomes outdated and obsolete, and users of these frameworks are constantly re-inventing the wheel.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/background/ml_explosion/ml_framework_evolution.png?raw=true
   :align: center
   :width: 80%

If our desire is to provide a new framework which simultaneously supports all of the modern frameworks in a simple and scalable manner, then we must determine exactly where the common ground lies between them.

Finding common ground between the existing frameworks is essential in order to design a simple, scalable, and universal abstraction.

In the search for common ground, considering the language first, we can see that Python has become the clear front-runner.
Looking a little deeper at these python frameworks, we find that all of these follow the same core principles of operation, exposing almost identical core functional APIs, but with unique syntax and arguments.
There are only so many ways to manipulate a tensor, and unsurprisingly these fundamental tensor operations are consistent between frameworks.
The functions exposed by each framework follow very similar conventions to those of Numpy, first introduced in 2006.

A simple and scalable abstraction layer therefore presents itself.
The functional APIs of all existing ML frameworks are all cut from the same cloth, adhering to similar sets of functions but with differing syntax and semantics.

**Round Up**

Hopefully this has painted a clear picture of how many different ML frameworks have exploded onto the scene 🙂

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!