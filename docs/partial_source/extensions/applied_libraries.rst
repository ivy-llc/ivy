Applied Libraries
=================

In other parts of the overview, we have focused on the the Ivy framework itself.
Here, we explore how Ivy has been used to create a suite of libraries in various fields related to ML.
Aside from being useful tools for ML developers in any framework, these libraries are a perfect showcase of what is possible using Ivy!

Currently there are Ivy libraries for: Mechanics, 3D Vision, Robotics, Gym Environments and Differentiable Memory.
We run through some demos from these library now, and encourage you to pip install the libraries and run the demos yourself if you like what you see!

Ivy Mechanics
-------------

`Ivy Mechanics <https://github.com/unifyai/mech>`_ provides functions for conversions of orientation, pose, and positional representations, as well as transformations, and some other more applied functions.
The orientation module is the largest, with conversions to and from all Euler conventions, quaternions, rotation matrices, rotation vectors, and axis-angle representations.

For example, this demo shows the use of :code:`ivy_mech.target_facing_rotation_matrix`:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_mech/demo_a.gif?raw=true
   :align: center
   :width: 100%

This demo shows the use of :code:`ivy_mech.polar_to_cartesian_coords`:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_mech/demo_b.gif?raw=true
   :align: center
   :width: 100%

Ivy Vision
----------

`Ivy Vision <https://github.com/unifyai/vision>`_ focuses predominantly on 3D vision, with functions for image projections, co-ordinate frame transformation, forward warping, inverse warping, optical flow, depth generation, voxel grids, point clouds, and others.

For example, this demo shows the use of :code:`ivy_vision.coords_to_voxel_grid`:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_vision/voxel_grid_demo.gif?raw=true
   :align: center
   :width: 100%

This demo shows the use of :code:`ivy_vision.render_pixel_coords`:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_vision/point_render_demo.gif?raw=true
   :align: center
   :width: 100%

This demo shows Neural Radiance Fields (NeRF):

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_vision/nerf_demo.gif?raw=true
   :align: center
   :width: 100%


Ivy Robot
---------

`Ivy Robot <https://github.com/unifyai/robot>`_ provides functions and classes for gradient-based trajectory optimization and motion planning.
Classes are provided both for mobile robots and robot manipulators.

For example, this demo shows the use of :code:`ivy_robot.sample_spline_path` and :code:`ivy_robot.RigidMobile.sample_body` for gradient-based motion planning of a drone.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_robot/demo_a.gif?raw=true
   :align: center
   :width: 100%

This demo shows the use of :code:`ivy_robot.sample_spline_path` and :code:`ivy_robot.Manipulator.sample_links` for gradient-based motion planning of a robot manipulator:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_robot/demo_b.gif?raw=true
   :align: center
   :width: 100%

Ivy Gym
-------

`Ivy Gym <https://github.com/unifyai/gym>`_ provides differentiable implementations of the control environments provided by OpenAI Gym, as well as new “Swimmer” task which illustrates the simplicity of creating new tasks.
The differentiable nature of the environments means that the cumulative reward can be directly optimized in a supervised manner, without need for reinforcement learning.
Ivy Gym opens the door for intersectional research between supervised learning, trajectory optimization, and reinforcement learning.

For example, we show demos of each of the environments :code:`cartpole`, :code:`mountain_car`, :code:`pendulum`, :code:`reacher`, and :code:`swimmer` solved using direct trajectory optimization below.
We optimize for a specific starting state of the environment:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_gym/demo_a.gif?raw=true
   :align: center
   :width: 100%

We show demos of each of the environments :code:`cartpole`, :code:`mountain_car`, :code:`pendulum`, :code:`reacher`, and :code:`swimmer` solved using supervised learning via a policy network.
We train a policy which is conditioned on the environment state, and the starting state is then randomized between training steps:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_gym/demo_b.gif?raw=true
   :align: center
   :width: 100%

Ivy Memory
---------

`Ivy Memory <https://github.com/unifyai/memory>`_ provides differentiable memory modules, including learnt modules such as Neural Turing Machines (NTM), but also parameter-free modules such as End-to-End Egospheric Spatial Memory (ESM).

For example, in this demo we learn to copy a sequence using :code:`ivy_memory.NTM`:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_memory/demo_a.gif?raw=true
   :align: center
   :width: 100%

In this demo we create an egocentric 3D map of a room using :code:`ivy_memory.ESM`:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_memory/demo_b.gif?raw=true
   :align: center
   :width: 100%

**Round Up**

Hopefully this has given you an idea of what’s possible using Ivy’s collection of applied libraries, and more importantly, given you inspiration for what’s possible using Ivy 🙂

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
