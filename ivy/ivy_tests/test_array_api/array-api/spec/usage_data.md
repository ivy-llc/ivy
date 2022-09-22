(usage-data)=

# Usage Data

> Summary of existing array API design and usage.

## Introduction

With rare exception, technical standardization ("standardization") occurs neither in a vacuum nor from first principles. Instead, standardization finds its origins in two or more, sometimes competing, implementations differing in design and behavior. These differences introduce friction as those (e.g., downstream end-users and library authors) who operate at higher levels of abstraction must either focus on an implementation subset (e.g., only NumPy-like array libraries) or accommodate variation through increased complexity (e.g., if NumPy array, call method `.foo()`; else if Dask array, call method `.bar()`).

Standardization aspires to reduce this friction and is a process which codifies that which is common, while still encouraging experimentation and innovation. Through the process of standardization, implementations can align around a subset of established practices and channel development resources toward that which is new and novel. In short, standardization aims to thwart reinventing the proverbial wheel.

A foundational step in standardization is articulating a subset of established practices and defining those practices in unambiguous terms. To this end, the standardization process must approach the problem from two directions: **design** and **usage**. The former direction seeks to understand

-   current implementation design (APIs, names, signatures, classes, and objects)
-   current implementation semantics (calling conventions and behavior)

while the latter direction seeks to quantify API

-   consumers (e.g., which downstream libraries utilize an API?)
-   usage frequency (e.g., how often is an API consumed?)
-   consumption patterns (e.g., which optional arguments are provided and in what context?)

By analyzing both design and usage, the standardization process grounds specification decisions in empirical data and analysis.

## Design

To understand API design, standardization follows the following process.

- Identify a representative sample of commonly used Python array libraries (e.g., NumPy, Dask Array, CuPy, MXNet, JAX, TensorFlow, and PyTorch).
- Acquire public APIs (e.g., by analyzing module exports and scraping public documentation).
- Unify and standardize public API data representation for subsequent analysis.
- Extract commonalities and differences by analyzing the intersection and complement of available APIs.
- Derive a common API subset suitable for standardization (based on prevalence and ease of implementation), where such a subset may include attribute names, method names, and positional and keyword arguments.
- Leverage usage data to validate API need and to inform naming conventions, supported data types, and/or optional arguments.
- Summarize findings and provide tooling for additional analysis and exploration.

See the [`array-api-comparison`](https://github.com/data-apis/array-api-comparison)
repository for design data and summary analysis.

## Usage

To understand usage patterns, standardization follows the following process.

- Identify a representative sample of commonly used Python libraries ("downstream libraries") which consume the subset of array libraries identified during design analysis (e.g., pandas, Matplotlib, SciPy, Xarray, scikit-learn, and scikit-image).
- Instrument downstream libraries in order to record Python array API calls.
- Collect traces while running downstream library test suites.
- Transform trace data into structured data (e.g., as JSON) for subsequent analysis.
- Generate empirical APIs based on provided arguments and associated types, noting which downstream library called which empirical API and at what frequency.
- Derive a single inferred API which unifies the individual empirical API calling semantics.
- Organize API results in human-readable form as type definition files.
- Compare the inferred API to the documented API.

The following is an inferred API for `numpy.arange`. The docstring includes the number of lines of code that invoked this function for each downstream library when running downstream library test suites.

```python
def arange(
    _0: object,
    /,
    *_args: object,
    dtype: Union[type, str, numpy.dtype, None] = ...,
    step: Union[int, float] = ...,
    stop: int = ...,
):
    """
    usage.dask: 347
    usage.matplotlib: 359
    usage.pandas: 894
    usage.sample-usage: 4
    usage.scipy: 1173
    usage.skimage: 174
    usage.sklearn: 373
    usage.xarray: 666
    """
    ...
```

See the [`python-record-api`](https://github.com/data-apis/python-record-api) repository for source code, usage data, and analysis. To perform a similar analysis on additional downstream libraries, including those not publicly released, see the published PyPI [package](https://pypi.org/project/record_api/).

## Use in Decision-Making

Design and usage data support specification decision-making in the following ways.

- Validate user stories to ensure that proposals satisfy existing needs.
- Define scope to ensure that proposals address general array library design requirements (i.e., proposals must have broad applicability and be possible to implement with a reasonable amount of effort). 
- Inform technical design discussions to ensure that proposals are grounded in empirical data.