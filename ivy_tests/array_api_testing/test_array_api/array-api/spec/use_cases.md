(use-cases)=

# Use cases

Use cases inform the requirements for, and design choices made in, this array
API standard. This section first discusses what types of use cases are
considered, and then works out a few concrete use cases in more detail.

## Types of use cases

- Packages that depend on a specific array library currently, and would like
  to support multiple of them (e.g. for GPU or distributed array support, for
  improved performance, or for reaching a wider user base).
- Writing new libraries/tools that wrap multiple array libraries.
- Projects that implement new types of arrays with, e.g., hardware-specific
  optimizations or auto-parallelization behavior, and need an API to put on
  top that is familiar to end users.
- End users that want to switch from one library to another without learning
  about all the small differences between those libraries.


## Concrete use cases

- {ref}`use-case-scipy`
- {ref}`use-case-einops`
- {ref}`use-case-xtensor`
- {ref}`use-case-numba`


(use-case-scipy)=

### Use case 1: add hardware accelerator and distributed support to SciPy

When surveying a representative set of advanced users and research software
engineers in 2019 (for [this NSF proposal](https://figshare.com/articles/Mid-Scale_Research_Infrastructure_-_The_Scientific_Python_Ecosystem/8009441)),
the single most common pain point brought up about SciPy was performance.

SciPy heavily relies on NumPy (its only non-optional runtime dependency).
NumPy provides an array implementation that's in-memory, CPU-only and
single-threaded. Common performance-related wishes users have are:

- parallel algorithms (can be multi-threaded or multiprocessing based)
- support for distributed arrays (with Dask in particular)
- support for GPUs and other hardware accelerators (shortened to just "GPU"
  in the rest of this use case)

Some parallelism can be supported in SciPy, it has a `workers` keyword
(similar to scikit-learn's `n_jobs` keyword) that allows specifying to use
parallelism in some algorithms. However SciPy itself will not directly start
depending on a GPU or distributed array implementation, or contain (e.g.)
CUDA code - that's not maintainable given the resources for development.
_However_, there is a way to provide distributed or GPU support. Part of the
solution is provided by NumPy's "array protocols" (see [gh-1](https://github.com/data-apis/array-api/issues/1)), that allow
dispatching to other array implementations. The main problem then becomes how
to know whether this will work with a particular distributed or GPU array
implementation - given that there are zero other array implementations that
are even close to providing full NumPy compatibility - without adding that
array implementation as a dependency.

It's clear that SciPy functionality that relies on compiled extensions (C,
C++, Cython, Fortran) directly can't easily be run on another array library
than NumPy (see [C API](design_topics/C_API.md) for more details about this topic). Pure Python
code can work though. There's two main possibilities:

1. Testing with another package, manually or in CI, and simply provide a list
   of functionality that is found to work. Then make ad-hoc fixes to expand
   the set that works.
2. Start relying on a well-defined subset of the NumPy API (or a new
   NumPy-like API), for which compatibility is guaranteed.

Option (2) seems strongly preferable, and that "well-defined subset" is _what
an API standard should provide_. Testing will still be needed, to ensure there
are no critical corner cases or bugs between array implementations, however
that's then a very tractable task.

As a concrete example, consider the spectral analysis functions in `scipy.signal`.
All of those functions (e.g., `periodogram`, `spectrogram`, `csd`, `welch`, `stft`,
`istft`) are pure Python - with the exception of `lombscargle` which is ~40
lines of Cython - and uses NumPy function calls, array attributes and
indexing. The beginning of each function could be changed to retrieve the
module that implements the array API standard for the given input array type,
and then functions from that module could be used instead of NumPy functions.

If the user has another array type, say a CuPy or PyTorch array `x` on their
GPU, doing:
```
from scipy import signal

signal.welch(x)
```
will result in:
```
# For CuPy
ValueError: object __array__ method not producing an array

# For PyTorch
TypeError: can't convert cuda:0 device type tensor to numpy.
```
and therefore the user will have to explicitly convert to and from a
`numpy.ndarray` (which is quite inefficient):
```
# For CuPy
x_np = cupy.asnumpy(x)
freq, Pxx = (cupy.asarray(res) for res in signal.welch(x_np))

# For PyTorch
x_np = x.cpu().numpy()
# Note: ends up with tensors on CPU, may still have to move them back
freq, Pxx = (torch.tensor(res) for res in signal.welch(x_np))
```
This code will look a little different for each array library. The end goal
here is to be able to write this instead as:
```
freq, Pxx = signal.welch(x)
```
and have `freq`, `Pxx` be arrays of the same type and on the same device as `x`.

```{note}

This type of use case applies to many other libraries, from scikit-learn
and scikit-image to domain-specific libraries like AstroPy and
scikit-bio, to code written for a single purpose or user.
```

(use-case-einops)=

### Use case 2: simplify einops by removing the backend system

[einops](https://github.com/arogozhnikov/einops) is a library that provides flexible tensor operations and supports many array libraries (NumPy, TensorFlow, PyTorch, CuPy, MXNet, JAX).
Most of the code in `einops` is:

- [einops.py](https://github.com/arogozhnikov/einops/blob/master/einops/einops.py)
  contains the functions it offers as public API (`rearrange`, `reduce`, `repeat`).
- [_backends.py](https://github.com/arogozhnikov/einops/blob/master/einops/_backends.py)
  contains the glue code needed to support that many array libraries.

The amount of code in each of those two files is almost the same (~550 LoC each).
The typical pattern in `einops.py` is:
```
def some_func(x):
    ...
    backend = get_backend(x)
    shape = backend.shape(x)
    result = backend.reduce(x)
    ...
```
With a standard array API, the `_backends.py` glue layer could almost completely disappear,
because the purpose it serves (providing a unified interface to array operations from each
of the supported backends) is already addressed by the array API standard.
Hence the complete `einops` code base could be close to 50% smaller, and easier to maintain or add to.

```{note}

Other libraries that have a similar backend system to support many array libraries
include [TensorLy](https://github.com/tensorly/tensorly), the (now discontinued)
multi-backend version of [Keras](https://github.com/keras-team/keras),
[Unumpy](https://github.com/Quansight-Labs/unumpy) and
[EagerPy](https://github.com/jonasrauber/eagerpy). Many end users and
organizations will also have such glue code - it tends to be needed whenever
one tries to support multiple array types in a single API.
```


(use-case-xtensor)=

### Use case 3: adding a Python API to xtensor

[xtensor](https://github.com/xtensor-stack/xtensor) is a C++ array library
that is NumPy-inspired and provides lazy arrays. It has Python (and Julia and R)
bindings, however it does not have a Python array API.

Xtensor aims to follow NumPy closely, however it only implements a subset of functionality
and documents some API differences in
[Notable differences with NumPy](https://xtensor.readthedocs.io/en/latest/numpy-differences.html).

Note that other libraries document similar differences, see for example
[this page for JAX](https://jax.readthedocs.io/en/latest/jax.numpy.html) and
[this page for TensorFlow](https://www.tensorflow.org/guide/tf_numpy).

Each time an array library author designs a new API, they have to choose (a)
what subset of NumPy makes sense to implement, and (b) where to deviate
because NumPy's API for a particular function is suboptimal or the semantics
don't fit their execution model.

This array API standard aims to provide an API that can be readily adopted,
without having to make the above-mentioned choices.

```{note}

XND is another array library, written in C, that still needs a Python API.
Array implementations in other languages are often in a similar situation,
and could translate this array API standard 1:1 to their language.
```


(use-case-numba)=

### Use case 4: make JIT compilation of array computations easier and more robust

[Numba](https://github.com/numba/numba) is a Just-In-Time (JIT) compiler for
numerical functions in Python; it is NumPy-aware. [PyPy](https://pypy.org)
is an implementation of Python with a JIT at its core; its NumPy support relies
on running NumPy itself through a compatibility layer (`cpyext`), while a
previous attempt to implement NumPy support directly was unsuccessful.

Other array libraries may have an internal JIT (e.g., TensorFlow, PyTorch,
JAX, MXNet) or work with an external JIT like
[XLA](https://www.tensorflow.org/xla) or [VTA](https://tvm.apache.org/docs/vta/index.html).

Numba currently has to jump through some hoops to accommodate NumPy's casting rules
and may not attain full compatibility with NumPy in some cases - see, e.g.,
[this](https://github.com/numba/numba/issues/4749) or
[this](https://github.com/numba/numba/issues/5907) example issue regarding (array) scalar
return values.

An [explicit suggestion from a Numba developer](https://twitter.com/esc___/status/1295389487485333505)
for this array API standard was:

> for JIT compilers (e.g. Numba) it will be important, that the type of the
  returned value(s) depends only on the *types* of the input but not on the
  *values*.

A concrete goal for this use case is to have better matching between
JIT-compiled and non-JIT execution. Here is an example from the Numba code
base, the need for which should be avoided in the future:

```
def check(x, y):
    got = cfunc(x, y)
    np.testing.assert_array_almost_equal(got, pyfunc(x, y))
    # Check the power operation conserved the input's dtype
    # (this is different from Numpy, whose behaviour depends on
    #  the *values* of the arguments -- see PyArray_CanCastArrayTo).
    self.assertEqual(got.dtype, x.dtype)
```