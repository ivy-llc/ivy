import io
import os
import warnings
import pickle
import numpy as np

MAGIC_PREFIX = b"\x93NUMPY"
MAGIC_LEN = len(MAGIC_PREFIX) + 2
GROWTH_AXIS_MAX_DIGITS = 21  # = len(str(8*2**64-1)) hypothetical int1 dtype
ARRAY_ALIGN = 64  # plausible values are powers of 2 between 16 and 4096
_header_size_info = {
    (1, 0): ("<H", "latin1"),
    (2, 0): ("<I", "latin1"),
    (3, 0): ("<I", "utf8"),
}


def _savez(file, args, kwds):
    import zipfile

    if not hasattr(file, "write"):
        file = os.fspath(file)
        if not file.endswith(".npz"):
            file = file + ".npz"
    namedict = kwds
    for i, val in enumerate(args):
        key = "arr_%d" % i
        if key in namedict.keys():
            raise ValueError("Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val

    zipf = zipfile_factory(file, mode="w", compression=zipfile.ZIP_STORED)

    for key, val in namedict.items():
        fname = key + ".npy"
        val = np.asanyarray(val)
        with zipf.open(fname, "w", force_zip64=True) as fid:
            write_array(fid, val)

    zipf.close()


def zipfile_factory(file, *args, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64, and the `file` argument can accept file, str, or
    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile
    constructor.
    """
    if not hasattr(file, "read"):
        file = os.fspath(file)
    import zipfile

    kwargs["allowZip64"] = True
    return zipfile.ZipFile(file, *args, **kwargs)


def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    """
    Write an array to an NPY file, including a header.

    If the array is neither C-contiguous nor Fortran-contiguous AND the
    file_like object is not a real file object, this function will have to
    copy data in memory.

    Parameters
    ----------
    fp : file_like object
        An open, writable file object, or similar object with a
        ``.write()`` method.
    array : ndarray
        The array to write to disk.
    version : (int, int) or None, optional
        The version number of the format. None means use the oldest
        supported version that is able to store the data.  Default: None
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: True
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass to pickle.dump, excluding
        'protocol'. These are only useful when pickling objects in object
        arrays on Python 3 to Python 2 compatible format.

    Raises
    ------
    ValueError
        If the array cannot be persisted. This includes the case of
        allow_pickle=False and array being an object array.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise various errors if the objects
        are not picklable.
    """
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)

    if array.itemsize == 0:
        buffersize = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024**2 // array.itemsize, 1)

    dtype_class = type(array.dtype)

    if array.dtype.hasobject or not dtype_class._legacy:
        # We contain Python objects so we cannot write out the data
        # directly.  Instead, we will pickle it out
        if not allow_pickle:
            if array.dtype.hasobject:
                raise ValueError(
                    "Object arrays cannot be saved when allow_pickle=False"
                )
            if not dtype_class._legacy:
                raise ValueError(
                    "User-defined dtypes cannot be saved when allow_pickle=False"
                )
        if pickle_kwargs is None:
            pickle_kwargs = {}
        pickle.dump(array, fp, protocol=3, **pickle_kwargs)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffersize,
                order="F",
            ):
                fp.write(chunk.tobytes("C"))
    else:
        if isfileobj(fp):
            array.tofile(fp)
        else:
            for chunk in np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffersize,
                order="C",
            ):
                fp.write(chunk.tobytes("C"))


def _check_version(version):
    if version not in [(1, 0), (2, 0), (3, 0), None]:
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))


def _write_array_header(fp, d, version=None):
    """
    Write the header for an array and returns the version used.

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    version : tuple or None
        None means use oldest that works. Providing an explicit version will
        raise a ValueError if the format does not allow saving this data.
        Default: None
    """
    header = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)

    # Add some spare space so that the array header can be modified in-place
    # when changing the array size, e.g. when growing it by appending data at
    # the end.
    shape = d["shape"]
    header += " " * (
        (GROWTH_AXIS_MAX_DIGITS - len(repr(shape[-1 if d["fortran_order"] else 0])))
        if len(shape) > 0
        else 0
    )

    if version is None:
        header = _wrap_header_guess_version(header)
    else:
        header = _wrap_header(header, version)
    fp.write(header)


def _wrap_header_guess_version(header):
    """Like `_wrap_header`, but chooses an appropriate version given the contents."""
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass

    try:
        ret = _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn(
            "Stored array in format 2.0. It can only beread by NumPy >= 1.9",
            UserWarning,
            stacklevel=2,
        )
        return ret

    header = _wrap_header(header, (3, 0))
    warnings.warn(
        "Stored array in format 3.0. It can only be read by NumPy >= 1.17",
        UserWarning,
        stacklevel=2,
    )
    return header


def _wrap_header(header, version):
    """Take a stringified header, and attaches the prefix and padding to it."""
    import struct

    assert version is not None
    fmt, encoding = _header_size_info[version]
    header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = ARRAY_ALIGN - ((MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN)
    try:
        header_prefix = magic(*version) + struct.pack(fmt, hlen + padlen)
    except struct.error:
        msg = "Header length {} too big for version={}".format(hlen, version)
        raise ValueError(msg) from None

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a
    # ARRAY_ALIGN byte boundary.  This supports memory mapping of dtypes
    # aligned up to ARRAY_ALIGN on systems like Linux where mmap()
    # offset must be page-aligned (i.e. the beginning of the file).
    return header_prefix + header + b" " * padlen + b"\n"


def magic(major, minor):
    """
    Return the magic string for the given file format version.

    Parameters
    ----------
    major : int in [0, 255]
    minor : int in [0, 255]

    Returns
    -------
    magic : str

    Raises
    ------
    ValueError if the version cannot be formatted.
    """
    if major < 0 or major > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if minor < 0 or minor > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return MAGIC_PREFIX + bytes([major, minor])


def header_data_from_array_1_0(array):
    """
    Get the dictionary of header metadata from a numpy.ndarray.

    Parameters
    ----------
    array : numpy.ndarray

    Returns
    -------
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """
    d = {"shape": array.shape}
    if array.flags.c_contiguous:
        d["fortran_order"] = False
    elif array.flags.f_contiguous:
        d["fortran_order"] = True
    else:
        # Totally non-contiguous data. We will have to make it C-contiguous
        # before writing. Note that we need to test for C_CONTIGUOUS first
        # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
        d["fortran_order"] = False

    d["descr"] = dtype_to_descr(array.dtype)
    return d


def dtype_to_descr(dtype):
    """
    Get a serializable descriptor from the dtype.

    The .descr attribute of a dtype object cannot be round-tripped through
    the dtype() constructor. Simple types, like dtype('float32'), have
    a descr which looks like a record array with one field with '' as
    a name. The dtype() constructor interprets this as a request to give
    a default name.  Instead, we construct descriptor that can be passed to
    dtype().

    Parameters
    ----------
    dtype : dtype
        The dtype of the array that will be written to disk.

    Returns
    -------
    descr : object
        An object that can be passed to `numpy.dtype()` in order to
        replicate the input dtype.
    """
    # NOTE: that drop_metadata may not return the right dtype e.g. for user
    #       dtypes.  In that case our code below would fail the same, though.
    new_dtype = drop_metadata(dtype)
    if new_dtype is not dtype:
        warnings.warn(
            "metadata on a dtype is not saved to an npy/npz. "
            "Use another format (such as pickle) to store it.",
            UserWarning,
            stacklevel=2,
        )
    if dtype.names is not None:
        # This is a record array. The .descr is fine.  XXX: parts of the
        # record array with an empty name, like padding bytes, still get
        # fiddled with. This needs to be fixed in the C implementation of
        # dtype().
        return dtype.descr
    elif not type(dtype)._legacy:
        # this must be a user-defined dtype since numpy does not yet expose any
        # non-legacy dtypes in the public API
        #
        # non-legacy dtypes don't yet have __array_interface__
        # support. Instead, as a hack, we use pickle to save the array, and lie
        # that the dtype is object. When the array is loaded, the descriptor is
        # unpickled with the array and the object dtype in the header is
        # discarded.
        #
        # a future NEP should define a way to serialize user-defined
        # descriptors and ideally work out the possible security implications
        warnings.warn(
            "Custom dtypes are saved as python objects using the "
            "pickle protocol. Loading this file requires "
            "allow_pickle=True to be set.",
            UserWarning,
            stacklevel=2,
        )
        return "|O"
    else:
        return dtype.str


def drop_metadata(dtype, /):
    """
    Return the dtype unchanged if it contained no metadata or a copy of the dtype if it
    (or any of its structure dtypes) contained metadata.

    This utility is used by `np.save` and `np.savez` to drop metadata before
    saving.

    .. note::

        Due to its limitation this function may move to a more appropriate
        home or change in the future and is considered semi-public API only.

    .. warning::

        This function does not preserve more strange things like record dtypes
        and user dtypes may simply return the wrong thing.  If you need to be
        sure about the latter, check the result with:
        ``np.can_cast(new_dtype, dtype, casting="no")``.
    """
    if dtype.fields is not None:
        found_metadata = dtype.metadata is not None

        names = []
        formats = []
        offsets = []
        titles = []
        for name, field in dtype.fields.items():
            field_dt = drop_metadata(field[0])
            if field_dt is not field[0]:
                found_metadata = True

            names.append(name)
            formats.append(field_dt)
            offsets.append(field[1])
            titles.append(None if len(field) < 3 else field[2])

        if not found_metadata:
            return dtype

        structure = dict(
            names=names,
            formats=formats,
            offsets=offsets,
            titles=titles,
            itemsize=dtype.itemsize,
        )

        # NOTE: Could pass (dtype.type, structure) to preserve record dtypes...
        return np.dtype(structure, align=dtype.isalignedstruct)
    elif dtype.subdtype is not None:
        # subarray dtype
        subdtype, shape = dtype.subdtype
        new_subdtype = drop_metadata(subdtype)
        if dtype.metadata is None and new_subdtype is subdtype:
            return dtype

        return np.dtype((new_subdtype, shape))
    else:
        # Normal unstructured dtype
        if dtype.metadata is None:
            return dtype
        # Note that `dt.str` doesn't round-trip e.g. for user-dtypes.
        return np.dtype(dtype.str)


def isfileobj(f):
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        # BufferedReader/Writer may raise OSError when
        # fetching `fileno()` (e.g. when wrapping BytesIO).
        f.fileno()
        return True
    except OSError:
        return False
