(future-API-evolution)=

# Future API standard evolution

## Scope extensions

Proposals for scope extensions in a future version of the API standard will follow
the process documented at https://github.com/data-apis/governance/blob/master/process_document.md

In summary, proposed new APIs go through several maturity stages, and will only be
accepted in a future version of this API standard once they have reached the "Final"
maturity stage, which means multiple array libraries have compliant implementations
and real-world experience from use of those implementations is available.


## Backwards compatibility

Functions, objects, keywords and specified behavior are added to this API standard
only if those are already present in multiple existing array libraries, and if there is
data that those APIs are used. Therefore it is highly unlikely that future versions
of this standard will make backwards-incompatible changes.

The aim is for future versions to be 100% backwards compatible with older versions.
Any exceptions must have strong rationales and be clearly documented in the updated
API specification.


(api-versioning)=

## Versioning

This API standard uses the following versioning scheme:

- The version is date-based, in the form `yyyy.mm` (e.g., `2020.12`).
- The version shall not include a standard way to do `alpha`/`beta`/`rc` or
  `.post`/`.dev` type versions.
  _Rationale: that's for Python packages, not for a standard._
- The version must be made available at runtime via an attribute
  `__array_api_version__` by a compliant implementation, in `'yyyy.mm'` format
  as a string, in the namespace that implements the API standard.
  _Rationale: dunder version strings are the standard way of doing this._

No utilities for dealing with version comparisons need to be provided; given
the format simple string comparisons with Python operators (`=-`, `<`, `>=`,
etc.) will be enough.

```{note}

Rationale for the `yyyy.mm` versioning scheme choice:
the API will be provided as part of a library, which already has a versioning
scheme (typically PEP 440 compliant and in the form `major.minor.bugfix`),
and a way to access it via `module.__version__`. The API standard version is
completely independent from the package version. Given the standardization
process, it resembles a C/C++ versioning scheme (e.g. `C99`, `C++14`) more
than Python package versioning.
```

The frequency of releasing a new version of an API standard will likely be at
regular intervals and on the order of one year, however no assumption on
frequency of new versions appearing must be made.