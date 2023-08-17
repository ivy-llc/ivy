Gradients
=========

.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`gradients channel`: https://discord.com/channels/799879767196958751/1000043921633722509

Overview
--------

Gradients are a crucial aspect of all modern deep learning workflows. 
Different frameworks provide different APIs for gradient computation and there were a few considerations to be made while building a unified gradients API in Ivy.
There are a number of functions added in ivy to allow gradient computation, but we'll mainly focus on the most commonly used and the most general function :func:`ivy.execute_with_gradients`.
This is because the other gradient functions such as :func:`ivy.value_and_grad` and :func:`ivy.grad` can be considered as providing a subset of the functionality that :func:`ivy.execute_with_gradients` provides.

Example Usage of the Gradient API
---------------------------------

The :func:`ivy.execute_with_gradients` function signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following is pseudo function signature for the :func:`ivy.execute_with_gradients` function,

.. code-block:: python
    
    def execute_with_gradients (
        func : Callable,
        xs : Any arbitrary nest,
        xs_grad_idxs : Input indices,
        ret_grad_idxs : Output indices,
    ) : 
        return func_ret, grads

The :code:`func` in the input can be any user-defined function that returns a single scalar or any arbitrary nest of scalars.
By scalars, we are referring to zero-dimensional arrays.

So for example, following are some valid outputs by the :code:`func`,

.. code-block:: python
    
    ivy.array(12.)
    
    # OR

    ivy.Container(
        a=ivy.array(12.), 
        b=ivy.Container(
            c=ivy.array(15.),
            d=ivy.array(32.)
        )
    )

    # OR

    [ivy.array(25.), {'x': (ivy.array(21.), ivy.array(11.))}, (ivy.array(9.),)]

:code:`xs` can be any arbitrary nest of arrays and refers to the inputs passed to the :code:`func`, so we suggest designing your :code:`func` based on what inputs you pass in :code:`xs`.
The arrays in :code:`xs` can contain any arbitrary number of dimensions, the only constraint is on the output of the :code:`func` as explained above.

The :code:`xs_grad_idxs` and :code:`ret_grad_idxs` are intended to provide more control over the arrays gradients are computed with.
:code:`xs_grad_idxs` accepts the indices of the input arrays to compute gradients for, and :code:`ret_grad_idxs` accepts the indices of the output arrays to compute gradients with respect to.

An example using :func:`ivy.execute_with_gradients`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def func(xs) :
        return ivy.mean(xs[0] + xs[1].b)

    x = ivy.array([1., 2., 3.])
    x = ivy.Container(a=x, b=x)
    y = ivy.array([4., 5., 6.])
    y = ivy.Container(b=y, c=x)
    xs = [x, y]

    ret, grads = ivy.execute_with_gradients(
        func, 
        xs, 
        xs_grad_idxs=[[0]],
        ret_grad_idxs=[["a"]]
    )


Custom Gradient Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

There are various scenarios where users may want to define custom gradient computation rules for their functions.
Some of these are numerical stability, smoothing and clipping of the computed gradients.
Ivy provides the :func:`ivy.bind_custom_gradient_function` function to allow users to bind custom gradient computation logic to their functions.

Following is an example of usage of :func:`ivy.bind_custom_gradient_function`,

.. code-block:: python

    import ivy

    ivy.set_backend("torch")
    x = ivy.array(50.0)
    inter_func = lambda x: ivy.log1p(ivy.exp(x))

    # args –> ((xs, ret), upstream)
    def custom_grad_fn(*args):
        args1 = (1 - 10 / (1 + args[0][0]))
        return (args[1] * args)

    inter_func = ivy.bind_custom_gradient_function(
    inter_func, custom_grad_fn
    )
    func = lambda x: ivy.sum(inter_func(x) ** 2)

    ret, grad = ivy.execute_with_gradients(func, x)

The :code:`custom_grad_fn` here accepts :code:`*args` which has the structure :code:`((xs, ret), upstream)` where,

* :code:`xs` is the input similar to the one accepted in :func:`ivy.execute_with_gradients`
* :code:`ret` is the output of the forward pass of the :func:`inter_func`
* :code:`upstream` refers to the previously computed gradients while back-propagating


Design of the Gradient API
--------------------------

Our policy on gradients
^^^^^^^^^^^^^^^^^^^^^^^

* The gradient API is fully-functional in ivy.
* There is no explicit variable class or any public-facing function for adding gradient support to an ivy.Array.
* The gradient functions in ivy implicitly convert all arrays to support gradient computation before computing gradients and detach all arrays after computing gradients.
* We don't retain any previously tracked computations in arrays by frameworks like torch for e.g. 
* This makes our gradient API disambiguous, flexible and easy to debug.
* Any framework-specific tracking of computations or variable classes should be handled in the corresponding frontends.

Gradient APIs of frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Frameworks and their gradient functions
   :widths: 25 25 50
   :header-rows: 1

   * - Framework
     - Common ways to Gradient Computation
   * - JAX
     - `jax.grad`, `jax.value_and_grad`, `jax.jacfwd`, `jax.jacrev`
   * - PyTorch
     - `torch.autograd.grad`, `torch.autograd.backward`
   * - TensorFlow
     - `tf.GradientTape`, `tf.gradients` (only in graph-mode)

General Structure of Backend-specific implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a high-level description of the steps followed backend-specific implementation of :func:`ivy.execute_with_gradients`:

#. Get Duplicate Index Chains : indices of arrays that share the same :code:`id`
#. Convert integer arrays to floats : only for ease of use. it's *not* recommended to pass integer arrays to gradient functions
#. Get relevant inputs : based on the :code:`xs_grad_idxs`, we collect the relevant inputs for gradient computation
#. Enable gradient support : we implicitly make use of framework-specific APIs to enable gradients in arrays. Ivy doesn't need to have an explicit variable class as the gradient API is fully functional
#. Compute Results : we do the forward pass by passing the input as it is to the function
#. Get relevant outputs : based on the :code:`ret_grad_idxs`, we collect the relevant outputs for gradient computation
#. Compute gradients : we make use of the framework-specific APIs to compute the gradients for the relevant outputs with respect to the relevant inputs
#. Handle duplicates : we explicitly handle duplicate instances using the index chains captured above as different frameworks treat duplicates differently
#. Post process and detach : finally, all computed gradients are updated to deal with :code:`NaN` and :code:`inf` and the input arrays are detached (i.e. gradient propagation is stopped)

Framework-specific Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* JAX treats duplicate arrays as distinct while computing gradients, so we need additional logic to replicate gradients computed w.r.t one array over all its duplicates.
* Gradients computed for functions with undefined results are inconsistent across backends (NaN, Inf, 0). We handle all these inconsistencies by returning 0 for all backends. So if you’re debugging gradients and find a 0, there’s a possibility that it was NaN or an Inf before computing.


**Round Up**

This should have hopefully given you a good feel for how the gradient API is implemented in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `gradients channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/riNddnTgDdk" class="video">
    </iframe>