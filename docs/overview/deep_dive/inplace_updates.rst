Inplace Updates
===============

.. _`backend setting`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`handle_out_argument`: https://github.com/unifyai/ivy/blob/dcfec8b85de3c422dc0ca1970d67cb620cae62a4/ivy/func_wrapper.py#L340
.. _`torch.tan`: https://pytorch.org/docs/stable/generated/torch.tan.html
.. _`numpy.tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`tf.math.tan`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`jax.numpy.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html?highlight=tan
.. _`presence of this attribute`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L341
.. _`by the backend function`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L372
.. _`handled by the wrapper`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L373
.. _`_wrap_fn`: https://github.com/unifyai/ivy/blob/6497b8a3d6b0d8aac735a158cd03c8f98eb288c2/ivy/container/wrapping.py#L69
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`ivy.reshape`: https://github.com/unifyai/ivy/blob/633eb420c5006a0a17c238bfa794cf5b6add8598/ivy/functional/ivy/manipulation.py#L418
.. _`ivy.astype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L46
.. _`ivy.asarray`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/creation.py#L114
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`inplace updates thread`: https://discord.com/channels/799879767196958751/1189906590166437938
.. _`example`: https://github.com/unifyai/ivy/blob/0ef2888cbabeaa8f61ce8aaea4f1175071f7c396/ivy/functional/ivy/layers.py#L169-L176


Inplace updates enable users to overwrite the contents of existing arrays with new data.
This enables much more control over the memory-efficiency of the program, preventing old unused arrays from being kept in memory for any longer than is strictly necessary.

The function :func:`ivy.inplace_update` enables explicit inplace updates.
:func:`ivy.inplace_update` is a *primary* function, and the backend-specific implementations for each backend are presented below.
We also explain the rationale for why each implementation is the way it is, and the important differences.

This is one particular area of the Ivy code where, technically speaking, the function :func:`ivy.inplace_update` will result in subtly different behaviour for each backend, unless the :code:`ensure_in_backend` flag is set to :code:`True`.

While :class:`ivy.Array` instances will always be inplace updated consistently, in some cases it is simply not possible to also inplace update the :class:`ivy.NativeArray` which :class:`ivy.Array` wraps, due to design choices made by each backend.

**NOTE:** Native inplace updates do not modify the dtype of the array being updated, as such the :code:`keep_input_dtype` flag should normally be set to :code:`True` such that inplace updating behavior is consistent between backends.

**JAX**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, JaxArray],
        val: Union[ivy.Array, JaxArray],
        /,
        *,
        ensure_in_backend: bool = False,
        keep_input_dtype: bool = False,
    ) -> ivy.Array:
        if ivy.is_array(x) and ivy.is_array(val):
            if ensure_in_backend:
                raise ivy.utils.exceptions.IvyException(
                    "JAX does not natively support inplace updates"
                )
            if keep_input_dtype:
                val = ivy.astype(val, x.dtype)
            (x_native, val_native), _ = ivy.args_to_native(x, val)
            if ivy.is_ivy_array(x):
                x.data = val_native
                # Handle view updates
                if ivy.exists(x._base):
                    base = x._base
                    base_idx = ivy.arange(base.size).reshape(base.shape)
                    for fn, args, kwargs, index in x._manipulation_stack:
                        kwargs["copy"] = True
                        base_idx = ivy.__dict__[fn](base_idx, *args, **kwargs)
                        base_idx = base_idx[index] if ivy.exists(index) else base_idx
                    base_flat = base.data.flatten()
                    base_flat = base_flat.at[base_idx.data.flatten()].set(
                        val_native.flatten()
                    )

                    base.data = base_flat.reshape(base.shape)

                    for ref in base._view_refs:
                        view = ref()
                        if ivy.exists(view) and view is not x:
                            _update_view(view, base)

                else:
                    for ref in x._view_refs:
                        view = ref()
                        if ivy.exists(view):
                            _update_view(view, x)
            else:
                raise ivy.utils.exceptions.IvyException(
                    "JAX does not natively support inplace updates"
                )
            return x
        else:
            return val

JAX **does not** natively support inplace updates, and so there is no way of actually inplace updating the :code:`JaxArray` instance :code:`x_native`.
Therefore, an inplace update is only performed on :class:`ivy.Array` instances provided in the input.

JAX functions also never returns views so additional logic is added to functionally support multiple variables referencing the same memory (further explained in a later section).

**NumPy**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, np.ndarray],
        val: Union[ivy.Array, np.ndarray],
        /,
        *,
        ensure_in_backend: bool = False,
        keep_input_dtype: bool = False,
    ) -> ivy.Array:
        ivy.utils.assertions.check_inplace_sizes_valid(x, val)
        if ivy.is_array(x) and ivy.is_array(val):
            if keep_input_dtype:
                val = ivy.astype(val, x.dtype)
            (x_native, val_native), _ = ivy.args_to_native(x, val)

            # make both arrays contiguous if not already
            if not x_native.flags.c_contiguous:
                x_native = np.ascontiguousarray(x_native)
            if not val_native.flags.c_contiguous:
                val_native = np.ascontiguousarray(val_native)

            if val_native.shape == x_native.shape:
                if x_native.dtype != val_native.dtype:
                    x_native = x_native.astype(val_native.dtype)
                np.copyto(x_native, val_native)
            else:
                x_native = val_native
            if ivy.is_ivy_array(x):
                x.data = x_native
            else:
                x = ivy.Array(x_native)
            return x
        else:
            return val

NumPy **does** natively support inplace updates, and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :class:`ivy.Array` instance, if provided in the input.

**TensorFlow**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, tf.Tensor],
        val: Union[ivy.Array, tf.Tensor],
        /,
        *,
        ensure_in_backend: bool = False,
        keep_input_dtype: bool = False,
    ) -> ivy.Array:
        if ivy.is_array(x) and ivy.is_array(val):
            if keep_input_dtype:
                val = ivy.astype(val, x.dtype)
            (x_native, val_native), _ = ivy.args_to_native(x, val)
            if _is_variable(x_native):
                x_native.assign(val_native)
                if ivy.is_ivy_array(x):
                    x.data = x_native
                else:
                    x = ivy.Array(x_native)
            elif ensure_in_backend:
                raise ivy.utils.exceptions.IvyException(
                    "TensorFlow does not support inplace updates of the tf.Tensor"
                )
            elif ivy.is_ivy_array(x):
                x.data = val_native
                # Handle view updates
                if ivy.exists(x._base):
                    base = x._base
                    base_idx = ivy.arange(base.size).reshape(base.shape)
                    for fn, args, kwargs, index in x._manipulation_stack:
                        kwargs["copy"] = True
                        base_idx = ivy.__dict__[fn](base_idx, *args, **kwargs)
                        base_idx = base_idx[index] if ivy.exists(index) else base_idx
                    base_flat = tf.reshape(base.data, -1)
                    base_flat = tf.tensor_scatter_nd_update(
                        base_flat,
                        tf.reshape(base_idx.data, (-1, 1)),
                        tf.reshape(val_native, -1),
                    )

                    base.data = tf.reshape(base_flat, base.shape)
                    for ref in base._view_refs:
                        view = ref()
                        if ivy.exists(view) and view is not x:
                            _update_view(view, base)
                else:
                    for ref in x._view_refs:
                        view = ref()
                        if ivy.exists(view):
                            _update_view(view, x)
            else:
                x = ivy.to_ivy(x_native)
            return x
        else:
            return val

TensorFlow **does not** natively support inplace updates for :class:`tf.Tensor` instances, and in such cases so there is no way of actually inplace updating the :class:`tf.Tensor` instance :code:`x_native`.
However, TensorFlow **does** natively support inplace updates for :class:`tf.Variable` instances.
Therefore, if :code:`x_native` is a :class:`tf.Variable`, then :code:`x_native` is updated inplace with :code:`val_native`.
Irrespective of whether the native array is a :class:`tf.Tensor` or a :class:`tf.Variable`, an inplace update is then also performed on the :class:`ivy.Array` instance, if provided in the input.

TensorFlow functions also never returns views so additional logic is added to functionally support multiple variables referencing the same memory (further explained in a later section).

**PyTorch**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, torch.Tensor],
        val: Union[ivy.Array, torch.Tensor],
        /,
        *,
        ensure_in_backend: bool = False,
        keep_input_dtype: bool = False,
    ) -> ivy.Array:
        ivy.utils.assertions.check_inplace_sizes_valid(x, val)
        if ivy.is_array(x) and ivy.is_array(val):
            if keep_input_dtype:
                val = ivy.astype(val, x.dtype)
            (x_native, val_native), _ = ivy.args_to_native(x, val)
            if is_variable(x_native):
                x_native.data = val_native
            else:
                x_native[()] = val_native
            if ivy.is_ivy_array(x):
                x.data = x_native
                _update_torch_views(x)
            else:
                x = ivy.to_ivy(x_native)
            if ensure_in_backend:
                x._data = val_native
            return x
        else:
            return val

PyTorch **does** natively support inplace updates, and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :class:`ivy.Array` instance, if provided in the input.

PyTorch also supports views for most manipulation and indexing operations as with NumPy but it lacks that functionality with a few functions such as :func:`flip`.
Additional logic had to be added to support view functionality for those functions (described in a section below).

The function :func:`ivy.inplace_update` is also *nestable*, meaning it can accept :class:`ivy.Container` instances in the input.
If an :class:`ivy.Container` instance is provided for the argument :code:`x`, then along with the arrays at all of the leaves, the container :code:`x` is **also** inplace updated, meaning that a new :class:`ivy.Container` instance is not created for the function return.

out argument
------------

Most functions in Ivy support inplace updates via the inclusion of a keyword-only :code:`out` argument.
This enables users to specify the array to which they would like the output of a function to be written.
This could for example be the input array itself, but can also be any other array of choice.

All Ivy functions which return a single array should support inplace updates via the :code:`out` argument.
The type hint of the :code:`out` argument is :code:`Optional[ivy.Array]`.
However, as discussed above, if the function is *nestable* then :class:`ivy.Container` instances are also supported.
:class:`ivy.Container` is omitted from the type hint in such cases, as explained in the `Function Arguments <function_arguments.rst>`_ section.

When the :code:`out` argument is unspecified, then the return is simply provided in a newly created :class:`ivy.Array` (or :class:`ivy.Container` if *nestable*).
However, when :code:`out` is specified, then the return is provided as an inplace update of the :code:`out` argument provided.
This can for example be the same as the input to the function, resulting in a simple inplace update of the input.

In the case of :class:`ivy.Array` return types, the :code:`out` argument is predominantly handled in `handle_out_argument`_.
As explained in the `Function Wrapping <function_wrapping.rst>`_ section, this wrapping is applied to every function with the :code:`@handle_out_argument` decorator dynamically during `backend setting`_.

**Primary Functions**

In the case of *primary* functions, `handle_out_argument`_ does not handle the backend-specific inplace updates in cases where the backend function being wrapped supports them directly, such as `torch.tan`_ and `numpy.tan`_, which both support the :code:`out` argument directly.
When implementing backend-specific functions, the attribute :code:`support_native_out` should be added to all functions which wrap a function in the backend supporting inplace updates directly.
`tf.math.tan`_ and `jax.numpy.tan`_ for example do **not** support inplace updates, and so the :code:`support_native_out` attribute should **not** be added to the :code:`tan` implementations.

The implementations of :func:`ivy.tan` for each backend are as follows.

**JAX** (no :code:`support_native_out` attribute):

.. code-block:: python

    def tan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
        return jnp.tan(x)

**NumPy** (includes :code:`support_native_out` attribute):

.. code-block:: python

    @_scalar_output_to_0d_array
    def tan(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.tan(x, out=out)


    tan.support_native_out = True

**TensorFlow** (no :code:`support_native_out` attribute):

.. code-block:: python

    def tan(
        x: Union[tf.Tensor, tf.Variable],
        /,
        *,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        return tf.tan(x)

**PyTorch** (includes :code:`support_native_out` attribute):

.. code-block:: python

    def tan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = _cast_for_unary_op(x)
        return torch.tan(x, out=out)

    tan.support_native_out = True

It's very important to ensure the :code:`support_native_out` attribute is not added to backend implementations that do not handle the :code:`out` argument, as the `presence of this attribute`_ dictates whether the argument should be handled `by the backend function`_ or `by the wrapper <function_wrapping.rst>`_.

This distinction only concerns how the inplace update is applied to the native array, which is operated upon directly by the backend.
If :code:`out` is specified in an Ivy function, then an inplace update is always **also** performed on the :class:`ivy.Array` instance itself, which is how :code:`out` is provided to the function originally.
The inplace update of this :class:`ivy.Array` is always `handled by the wrapper`_.

Alternatively, if :code:`out` is an :class:`ivy.Container`, then the inplace update is always handled by `_wrap_fn`_ in the container wrapping module.

**Special Case**

Take a function which has multiple possible "paths" through the code:

.. code-block:: python

    def cholesky(
        x: torch.Tensor, /, *, upper: bool = False, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not upper:
            return torch.linalg.cholesky(x, out=out)
        else:
            ret = torch.transpose(
                torch.linalg.cholesky(
                    torch.transpose(x, dim0=len(x.shape) - 1, dim1=len(x.shape) - 2)
                ),
                dim0=len(x.shape) - 1,
                dim1=len(x.shape) - 2,
            )
            if ivy.exists(out):
                return ivy.inplace_update(out, ret)
            return ret


    cholesky.support_native_out = True

Here we still have the :attr:`support_native_out` attribute since we want to take advantage of the native inplace update enabled by :func:`torch.linalg.cholesky` in the first condition.
However, in the :code:`else` statement, the last operation is :func:`torch.transpose` which does not support the :code:`out` argument, and so the native inplace update can't be performed by torch here.
This is why we need to call :func:`ivy.inplace_update` explicitly here, to ensure the native inplace update is performed, as well as the :class:`ivy.Array` inplace update.

Another case where we need to use :func:`ivy.inplace_update` with a function that has :attr:`support_native_out` is for the example of the :code:`torch` backend implementation of the :func:`ivy.remainder` function

.. code-block:: python

    def remainder(
        x1: Union[float, torch.Tensor],
        x2: Union[float, torch.Tensor],
        /,
        *,
        modulus: bool = True,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
        if not modulus:
            res = x1 / x2
            res_floored = torch.where(res >= 0, torch.floor(res), torch.ceil(res))
            diff = res - res_floored
            diff, x2 = ivy.promote_types_of_inputs(diff, x2)
            if ivy.exists(out):
                if out.dtype != x2.dtype:
                    return ivy.inplace_update(
                        out, torch.round(torch.mul(diff, x2)).to(out.dtype)
                    )
            return torch.round(torch.mul(diff, x2), out=out).to(x1.dtype)
        return torch.remainder(x1, x2, out=out).to(x1.dtype)


    remainder.support_native_out = True


Here, even though the :func:`torch.round` function natively supports the :code:`out` argument, in case the :code:`dtype` of the :code:`out` argument is different
from the :code:`dtype` of the result of the function, we need to use :func:`ivy.inplace_update`, while still trying to utilize the native :code:`out` argument whenever
the :code:`dtype` is the same for maximum possible extent of the native inplace update.

**Compositional Functions**

For *compositional* functions, the :code:`out` argument should **always** be handled in the compositional implementation, with no wrapping applied at all.
This is for a few reasons:

#. we need to show the :code:`out` argument in the compositional function signature, as this is the only function implementation in the codebase.
   Adding an argument unused in the implementation could cause some confusion.
#. generally, inplace updates are performed because memory management is an area of concern for the user.
   By handling the :code:`out` argument in the compositional implementation itself.
   We can maximize the memory efficiency of the function, using inplace updates in as many of the inner Ivy functions as possible.
#. this enables us to make use of backend-specific :code:`out` argument handling.

The second and third points are the most important points.

We'll use :func:`ivy.cross_entropy` as an example:


.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        epsilon: float = 1e-7,
        reduction: str = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return _reduce_loss(reduction, log_pred * true, axis, out=out)

By handling the :code:`out` argument in the function, we are able to get the benefits outlined above.
Firstly, the return of :func:`ivy.sum` is the same shape and type as the return of the entire function, and so we can also write this output to the :code:`out` argument inplace.
We can then subsequently overwrite the contents of :code:`out` again with the return of the :func:`ivy.negative` function.
This minimizes the number of arrays created during the execution of the function, which is generally the intention when specifying the :code:`out` argument.
Additionally, with a PyTorch backend, the :func:`ivy.negative` function defers to the :code:`out` argument of :func:`torch.negative` function directly, which is the most efficient inplace update possible, making use of backend-specific optimizations.

If we had instead simply used the wrapper `handle_out_argument`_, then we would not leverage any of these benefits, and instead simply call :func:`ivy.inplace_update` at the very end of the function call.

For some compositional functions, the internal function which generates the final return value does not itself support the :code:`out` argument.
For example, `ivy.multi_head_attention <https://github.com/unifyai/ivy/blob/2045db570d7977830681a7498a3c1045fb5bcc79/ivy/functional/ivy/layers.py#L165>`_ includes support for arbitrary functions passed in the input, including :code:`to_out_fn` which, if specified, is applied to the outputs before returning.
For such functions, the inplace update should just be performed using :func:`ivy.inplace_update` at the end of the function, like `so <https://github.com/unifyai/ivy/blob/2045db570d7977830681a7498a3c1045fb5bcc79/ivy/functional/ivy/layers.py#L254>`_.

Technically, this could be handled using the `handle_out_argument`_ wrapping, but we opt to implement this in the compositional function itself, due to point 1 mentioned above.

**Mixed Functions**

As explained in the `Function Types <function_types.rst>`_ section, *mixed* functions can effectively behave as either compositional or primary functions, depending on the backend that is selected. We must add the :code:`handle_out_argument` to the :code:`add_wrappers`key of
the :code:`mixed_backend_wrappers` attribute so that the decorator gets added to the primary implementation when the backend is set. Here's an `example`_ from the linear function.


copy argument
-------------

As well as the :code:`out` argument, many also support the :code:`copy` argument.
The functions with support for the :code:`copy` argument are either in the `Array API Standard`_, and the standard mandates the inclusion of :code:`copy` in each case.
Or they are expected to return views with specific backends (hence being decorated with the :code:`@handle_view` wrapper) and the :code:`copy` is added to allow a way to prevent views from being created.

The :code:`copy` argument dictates whether a new copy should be created, or whether the input array should be updated inplace.
When :code:`copy` is not specified explicitly, then an inplace update is performed with the same behaviour as :code:`copy=False`.
Setting :code:`copy=False` is equivalent to passing :code:`out=input_array`.
If only one of :code:`copy` or :code:`out` is specified, then this specified argument is given priority.
If both are specified, then priority is given to the more general :code:`out` argument.
As with the :code:`out` argument, the :code:`copy` argument is also handled `by the wrapper <function_wrapping.rst>`_.


Views
------------

Many functions in NumPy and PyTorch return views instead of copies, these functions are mostly manipulation routines or indexing routines.
Views are arrays which access the same data buffer as another array but view it with different metadata like :code:`stride`.
More information about these arrays can be found in `NumPy's documentation <https://numpy.org/doc/stable/user/basics.copies.html>`_.
This essentially means that any inplace update on the original array or any of its views will cause all the other views to be updated as well since they reference the same memory buffer.

We want to keep supporting NumPy and PyTorch inplace updates whenever we can and superset backend behaviour, however it is not trivial to replicate this in JAX and TensorFlow.
The main reason is because these frameworks do not natively support inplace updates so even if multiple native arrays are referencing the same memory buffer, you would never be able to update it once for all of them.
Therefore views and their updates must be tracked through Ivy and extra logic has been added to update views in the case an inplace update happens to any array which is meant to be referencing the same memory.
We call views tracked and updated by Ivy functional views as they work with a functional paradigm.

What functions return views is mostly dictated by NumPy since it has the most expansive support for them, any function which returns views in NumPy or PyTorch should be decorated with the :code:`@handle_view` wrapper, except :func:`get_item` which has it's own :code:`@handle_view_indexing` wrapper.
Every function with this wrapper should also have a :code:`copy` argument such that Ivy maintains a way to prevent views from being created if necessary.
What that wrapper does is update a few :class:`ivy.Array` attributes which help keep track of views, how they were created, and which arrays should be updated together.
These attributes are then used in the :func:`ivy.inplace_update` to update all the arrays which are meant to be referencing the same memory, at least to NumPy's standard.
Of course, these are normally only used with a JAX and TensorFlow backend since NumPy and PyTorch natively update their views and Ivy does not need to do any extra handling except for a few functions where PyTorch fails to return views when NumPy does.
The functions currently implemented in the Ivy API where PyTorch fails to return views at the time of writing are :func:`ivy.flip`, :func:`ivy.rot90`, :func:`ivy.flipud`, :func:`ivy.fliplr`.
In the case one of those functions is used with a Pytorch backend, additional logic has been added to make the returns of those functions behave as views of the original that made them.

Here's a brief description of the additional attributes added to :class:`ivy.Array` and their usage:

#. Base (:code:`._base`): the original array being referenced (array all views stem from)
#. Manipulation stack (:code:`._manipulation_stack`): store of operations that were done on the original to get to the current shape (manipulation or indexing)
#. Reference stack :code:`._view_refs`: Weak references to the arrays that reference the original as view, only populated for base arrays.
#. PyTorch Base (:code:`._torch_base`): Keeps track of functional view (array created from the listed functions above) that made it, otherwise stores original array
#. PyTorch reference stack (:code:`._torch_view_refs`): Functional views referencing this array in its PyTorch base, only populated for original arrays or functional views.
#. PyTorch manipulation cache (:code:`._torch_manipulation`): Tuple storing array or view and function which made the functional view, only populated for functional views

.. note::
    Parts of an arrays metadata like :code:`stride` are attributed to the low-level memory layout of arrays while views in :code:`ivy` operate at a higher level of abstraction.
    As a result, :func:`ivy.strides` isn't guaranteed to produce an output reflective of the underlying memory layout if the :class:`ivy.Array` passed in is a view (or in other words has a :code:`_base`).

Here's a brief description of how the :code:`@handle_view` wrapper populates these attributes:

#. When an array is made using a function decorated by this wrapper its base becomes the array that made it, or if the array that made it is also a view, its base.
#. The view is then added to the reference stack of the base array (weakly), the operation that created the array is also added to the manipulation stack of the array.
#. The way the PyTorch specific attributes are updated should be adequately explained above.

Here's a brief description of what happens during an inplace operation with a JAX and TensorFlow backend:

#. If the base is inplace updated, then it goes through all the arrays in the reference stack, and through their manipulation, then inplace updates every array respectively.
#. If a view gets inplace updated, an index array is created of the shape of the base array, which then is passed through the manipulation stack of the updated array.
#. The updated array and the index array are then flattened and they then update the original array by performing a scatter update on a flattened version of the original array, which then gets reshaped into the correct shape.
#. Then the all views stemming from the original are updated as described in the first point.

Here's a brief description of what happens during an inplace operation with a PyTorch backend:

#. The array being updated checks if it has a populated reference stack, if it does it inplace updates each functional view in the stack with the output of the stored function called with the array that made it.
   It then checks if the functional view has a reference stack and continues recursively until it reaches a point where it exhausts all reference stacks.
#. If the reference stack is empty or exhausted it checks if it has a manipulation stack.
   If populated it performs the reverse functional operation with itself as the input and inplace updates the view that made it (reverses the operation that made it).
   If the manipulation stack is empty or already exhausted it goes to the array’s PyTorch base and repeats the recursively until everything is exhausted and the base is None.
#. All other views are expected to be updated automatically through PyTorch's native view handling.

**Round Up**

This should have hopefully given you a good feel for inplace updates, and how these are handled in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `inplace updates thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/n8ko-Ig2eZ0" class="video">
    </iframe>
