"""
Utility functions
"""

import inspect
import os
import warnings
from collections.abc import Sequence
from itertools import combinations
from numbers import Number
from pathlib import Path

import numpy as np

from .config import nap_config


def convert_to_array(array, array_name):
    # Check if jax backend
    if get_backend() == "jax":
        from pynajax.utils import convert_to_jax_array

        return convert_to_jax_array(
            array, array_name, nap_config.suppress_conversion_warnings
        )
    else:
        return convert_to_numpy_array(array, array_name)


def convert_to_numpy_array(array, array_name):
    """Convert any array like object to numpy ndarray.

    Parameters
    ----------
    array : ArrayLike

    array_name : str
        Array name if RuntimeError is raised

    Returns
    -------
    numpy.ndarray
        Numpy array object

    Raises
    ------
    RuntimeError
        If input can't be converted to numpy array
    """
    if isinstance(array, Number):
        return np.array([array])
    elif isinstance(array, (list, tuple)):
        return np.array(array)
    elif isinstance(array, np.ndarray):
        return array
    elif is_array_like(array):
        return cast_to_numpy(array, array_name)
    else:
        raise RuntimeError(
            "Unknown format for {}. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.".format(
                array_name
            )
        )


def get_backend():
    """
    Return the current backend of pynapple. Possible backends are
    'numba' or 'jax'.
    """
    return nap_config.backend


def _get_terminal_size():
    """Helper to get terminal size for __repr__

    Returns
    -------
    tuple

    """
    cols = 100  # Default
    rows = 2
    try:
        cols, rows = os.get_terminal_size()
    except Exception:
        import shutil

        cols, rows = shutil.get_terminal_size()

    return (cols, rows)


def is_array_like(obj):
    """
    Check if an object is array-like.

    This function determines if an object has array-like properties.
    An object is considered array-like if it has attributes typically associated with arrays
    (such as `.shape`, `.dtype`, and `.ndim`), supports indexing, and is iterable.

    Parameters
    ----------
    obj : object
        The object to check for array-like properties.

    Returns
    -------
    bool
        True if the object is array-like, False otherwise.

    Notes
    -----
    This function uses a combination of checks for attributes (`shape`, `dtype`, `ndim`),
    indexability, and iterability to determine if the given object behaves like an array.
    It is designed to be flexible and work with various types of array-like objects, including
    but not limited to NumPy arrays and JAX arrays. However, it may not be full proof for all
    possible array-like types or objects that mimic these properties without being suitable for
    numerical operations.

    """
    # Check for array-like attributes
    has_shape = hasattr(obj, "shape")
    has_dtype = hasattr(obj, "dtype")
    has_ndim = hasattr(obj, "ndim")

    # Check for indexability (try to access the first element)

    try:
        obj[0]
        is_indexable = True
    except Exception:
        is_indexable = False

    if not is_indexable:
        if hasattr(obj, "__len__"):
            try:
                if len(obj) == 0:
                    is_indexable = True  # Could be an empty array
            except Exception:
                is_indexable = False

    # Check for iterable property
    try:
        iter(obj)
        is_iterable = True
    except Exception:
        is_iterable = False

    # not_tsd_type = not isinstance(obj, _AbstractTsd)

    return (
        has_shape
        and has_dtype
        and has_ndim
        and is_indexable
        and is_iterable
        # and not_tsd_type
    )


def cast_to_numpy(array, array_name):
    """
    Convert an input array-like object to a NumPy array.

    This function attempts to convert an input object to a NumPy array using `np.asarray`.
    If the input is not already a NumPy ndarray, it issues a warning indicating that a conversion
    has taken place and shows the original type of the input. This function is useful for
    ensuring compatibility with Numba operations in cases where the input might come from
    various array-like sources (for instance, jax.numpy.Array).

    Parameters
    ----------
    array : array_like
        The input object to convert. This can be any object that `np.asarray` is capable of
        converting to a NumPy array, such as lists, tuples, and other array-like objects,
        including those from libraries like JAX or TensorFlow that adhere to the array interface.
    array_name : str
        The name of the variable that we are converting, printed in the warning message.

    Returns
    -------
    ndarray
        A NumPy ndarray representation of the input `values`. If `values` is already a NumPy
        ndarray, it is returned unchanged. Otherwise, a new NumPy ndarray is created and returned.

    Warnings
    --------
    A warning is issued if the input `values` is not already a NumPy ndarray, indicating
    that a conversion has taken place and showing the original type of the input.

    """
    if (
        not isinstance(array, np.ndarray)
        and not nap_config.suppress_conversion_warnings
    ):
        original_type = type(array).__name__
        warnings.warn(
            f"Converting '{array_name}' to numpy.array. The provided array was of type '{original_type}'.",
            UserWarning,
        )
    return np.asarray(array)


def _check_time_equals(time_arrays):
    """
    Check if a list of time arrays are all equal.
    This is typically use to compare time index arrays or starts and ends of `IntervalSet`

    Parameters
    ----------
    time_arrays : list of arrays
        The time arrays to compare to each other

    Returns
    -------
    bool
        True if all equal else False

    """
    return all(
        map(
            lambda x: np.allclose(
                *x, rtol=0, atol=1 / (10**nap_config.time_index_precision)
            ),
            combinations(time_arrays, 2),
        )
    )


def _split_tsd(func, tsd, indices_or_sections, axis=0):
    """
    Wrappers of numpy split functions
    """
    if func in [np.split, np.array_split, np.vsplit] and axis == 0:
        out = func._implementation(tsd.values, indices_or_sections)
        index_list = np.split(tsd.index.values, indices_or_sections)
        return [
            tsd._define_instance(t, None, values=d) for t, d in zip(index_list, out)
        ]
    elif func in [np.dsplit, np.hsplit]:
        out = func._implementation(tsd.values, indices_or_sections)
        return [tsd._define_instance(tsd.index, None, values=d) for d in out]
    else:
        return func._implementation(tsd.values, indices_or_sections, axis)


def _concatenate_tsd(func, *args, **kwargs):
    """
    Wrappers of concatenation functions
    """
    arrays = []
    time_indexes = []
    time_supports = []
    time_origins = []
    nap_types = []
    columns = []
    nap_class = None

    if func == np.concatenate:  # search for axis
        if "axis" not in kwargs and len(args) >= 2:  # assume second arg is axis
            if isinstance(args[1], int):
                kwargs["axis"] = args[1]
            else:
                kwargs["axis"] = 0

    for arg in args[0]:
        if all(
            map(
                lambda x: hasattr(arg, x),
                ["values", "index", "time_support", "nap_class"],
            )
        ):
            arrays.append(arg.values)
            time_indexes.append(arg.index.values)
            time_supports.append(arg.time_support)
            time_origins.append(getattr(arg, "time_origin", None))
            nap_types.append(arg.nap_class)
            nap_class = arg.__class__
            if hasattr(arg, "columns"):
                columns.append(arg.columns)
        else:
            arrays.append(arg)

    output = func._implementation(arrays, **kwargs)

    # dimension increased in the first axis
    if output.shape[0] > arrays[0].shape[0]:
        if len(time_indexes) == len(arrays) and len(time_supports) == len(arrays):
            # check if time indexes can be concatenated
            new_index = np.hstack(time_indexes)
            if np.any(np.diff(new_index) <= 0):
                raise RuntimeError(
                    "The order of the time series indexes should be strictly increasing and non overlapping."
                )
            # Joining Time support
            time_support = time_supports[0]
            for support in time_supports[1:]:
                time_support = time_support.union(support)

            # Check time_origin compatibility
            shared_origin = time_origins[0] if time_origins else None
            for to in time_origins[1:]:
                if (shared_origin is None) != (to is None):
                    raise TypeError(
                        "Cannot concatenate synchronized and unsynchronized objects."
                    )
                if shared_origin is not None and to is not None and shared_origin != to:
                    raise ValueError(
                        "Objects have different time origins "
                        f"({shared_origin} vs {to})."
                    )

            new_kwargs = {"columns": columns[0]} if len(columns) else {}
            if shared_origin is not None:
                new_kwargs["time_origin"] = shared_origin

            return nap_class(
                t=new_index, d=output, time_support=time_support, **new_kwargs
            )
        else:
            return output
    # dimension increased in other axis
    else:
        if len(time_indexes) == 1:
            return nap_class(t=time_indexes[0], d=output, time_support=time_supports[0])
        else:
            time_equal = _check_time_equals(time_indexes)
            support_equal = _check_time_equals([x.values for x in time_supports])

            if time_equal and support_equal:
                new_kwargs = {}
                if len(columns):
                    new_kwargs = {"columns": np.hstack([c for c in columns])}
                    if len(new_kwargs["columns"]) != output.shape[1]:
                        new_kwargs = {}
                return args[0][0]._define_instance(
                    time_index=time_indexes[0],
                    time_support=time_supports[0],
                    values=output,
                    **new_kwargs,
                )  # Dropping metadata in this case
            else:
                if not time_equal and not support_equal:
                    msg = "Time indexes and time supports are not all equals up to pynapple precision. Returning numpy array!"
                elif not time_equal and support_equal:
                    msg = "Time indexes are not all equals up to pynapple precision. Returning numpy array!"
                else:
                    msg = "Time supports are not all equals up to pynapple precision. Returning numpy array!"

                warnings.warn(msg, stacklevel=2)
                return output


def concatenate(*objects):
    """Concatenate pynapple objects along the time axis.

    Supports Tsd, TsdFrame, TsdTensor, Ts, IntervalSet, and TsGroup.
    All inputs must be the same type and have compatible time_origin values.
    Timestamps are sorted automatically with a warning if originally unsorted.

    Parameters
    ----------
    *objects : pynapple objects
        Objects to concatenate. All must be the same type. Can also be passed
        as a single list/tuple of objects.

    Returns
    -------
    concatenated object
        Same type as input, with time_origin propagated.

    Raises
    ------
    ValueError
        If no objects are provided, types don't match, or time_origins differ.
    TypeError
        If mixing synchronized and unsynchronized objects.
    """
    # Allow passing a single list/tuple
    if len(objects) == 1 and isinstance(objects[0], (list, tuple)):
        objects = objects[0]

    if len(objects) == 0:
        raise ValueError("No objects to concatenate.")

    if len(objects) == 1:
        return objects[0]

    # Lazy imports to avoid circular dependencies
    from .interval_set import IntervalSet
    from .time_series import Ts, Tsd, TsdFrame, TsdTensor, _BaseTsd

    # Check all same type
    obj_type = type(objects[0])
    for i, obj in enumerate(objects[1:], 1):
        if type(obj) != obj_type:
            raise TypeError(
                f"All objects must be the same type. "
                f"Got {obj_type.__name__} and {type(obj).__name__} at position {i}."
            )

    # Check time_origin compatibility
    origins = [getattr(obj, "time_origin", None) for obj in objects]
    shared_origin = origins[0]
    for i, to in enumerate(origins[1:], 1):
        if (shared_origin is None) != (to is None):
            raise TypeError(
                "Cannot concatenate synchronized and unsynchronized objects."
            )
        if shared_origin is not None and to is not None and shared_origin != to:
            raise ValueError(
                f"Objects have different time origins ({shared_origin} vs {to})."
            )

    # Dispatch by type
    if isinstance(objects[0], IntervalSet):
        return _concatenate_interval_sets(objects, shared_origin)
    elif isinstance(objects[0], _BaseTsd) or isinstance(objects[0], Ts):
        return _concatenate_time_series(objects, shared_origin)
    else:
        # Try TsGroup via deferred import
        from .ts_group import TsGroup

        if isinstance(objects[0], TsGroup):
            return _concatenate_ts_groups(objects, shared_origin)
        else:
            raise TypeError(f"Cannot concatenate objects of type {obj_type.__name__}.")


def _concatenate_interval_sets(objects, shared_origin):
    """Concatenate IntervalSet objects (union)."""
    from .interval_set import IntervalSet

    all_starts = np.concatenate([obj.start for obj in objects])
    all_ends = np.concatenate([obj.end for obj in objects])
    return IntervalSet(start=all_starts, end=all_ends, time_origin=shared_origin)


def _concatenate_time_series(objects, shared_origin):
    """Concatenate time series objects along the time axis."""
    from .time_series import Ts, TsdFrame

    # Gather timestamps and data
    all_t = np.concatenate([obj.index.values for obj in objects])

    has_values = hasattr(objects[0], "values")
    if has_values:
        all_d = np.concatenate([obj.values for obj in objects], axis=0)
    else:
        all_d = None

    # Check if sorted, sort with warning if not
    if len(all_t) > 1 and np.any(np.diff(all_t) < 0):
        warnings.warn(
            "Timestamps are not sorted. Sorting them automatically.",
            stacklevel=3,
        )
        sort_idx = np.argsort(all_t, kind="stable")
        all_t = all_t[sort_idx]
        if all_d is not None:
            all_d = all_d[sort_idx]

    # Check for duplicate timestamps after sorting
    if len(all_t) > 1 and np.any(np.diff(all_t) == 0):
        raise ValueError(
            "Duplicate timestamps found after concatenation. "
            "Timestamps must be strictly increasing."
        )

    # Union time supports
    from .interval_set import IntervalSet

    time_support = objects[0].time_support
    for obj in objects[1:]:
        time_support = time_support.union(obj.time_support)

    # Build kwargs
    kwargs = {"time_origin": shared_origin}
    if isinstance(objects[0], TsdFrame) and hasattr(objects[0], "columns"):
        kwargs["columns"] = objects[0].columns

    # Construct result
    cls = type(objects[0])
    if all_d is not None:
        return cls(t=all_t, d=all_d, time_support=time_support, **kwargs)
    else:
        return cls(t=all_t, time_support=time_support, **kwargs)


def _concatenate_ts_groups(objects, shared_origin):
    """Concatenate TsGroup objects, merging overlapping keys."""
    from .ts_group import TsGroup

    merged_data = {}
    first_metadata = {}

    for obj in objects:
        for k in obj.keys():
            if k in merged_data:
                # Merge: concatenate the underlying Ts/Tsd
                existing = merged_data[k]
                new_obj = obj[k]
                merged_t = np.concatenate([existing.index.values, new_obj.index.values])
                if hasattr(existing, "values") and hasattr(new_obj, "values"):
                    merged_d = np.concatenate([existing.values, new_obj.values], axis=0)
                    # Sort if needed
                    if len(merged_t) > 1 and np.any(np.diff(merged_t) < 0):
                        sort_idx = np.argsort(merged_t, kind="stable")
                        merged_t = merged_t[sort_idx]
                        merged_d = merged_d[sort_idx]
                    merged_data[k] = type(existing)(t=merged_t, d=merged_d)
                else:
                    if len(merged_t) > 1 and np.any(np.diff(merged_t) < 0):
                        merged_t = np.sort(merged_t)
                    from .time_series import Ts

                    merged_data[k] = Ts(t=merged_t)
            else:
                merged_data[k] = obj[k]
                # Store first metadata for this key
                meta_cols = [
                    c for c in obj.metadata_columns if c != "rate"
                ]
                if meta_cols:
                    first_metadata[k] = {
                        c: obj._metadata[c][list(obj.keys()).index(k)]
                        for c in meta_cols
                    }

    # Build metadata dict
    if first_metadata:
        all_keys = sorted(merged_data.keys())
        meta_cols = list(next(iter(first_metadata.values())).keys())
        metadata = {
            c: [first_metadata.get(k, {}).get(c, None) for k in all_keys]
            for c in meta_cols
        }
    else:
        metadata = None

    # Union time supports
    time_support = objects[0].time_support
    for obj in objects[1:]:
        time_support = time_support.union(obj.time_support)

    return TsGroup(
        merged_data,
        time_support=time_support,
        bypass_check=True,
        metadata=metadata,
        time_origin=shared_origin,
    )


class _TsdFrameSliceHelper:
    def __init__(self, tsdframe):
        self.tsdframe = tsdframe

    def __getitem__(self, key):
        if hasattr(key, "__iter__") and not isinstance(key, str):
            for k in key:
                if k not in self.tsdframe.columns:
                    raise IndexError(str(k))
            index = self.tsdframe.columns.get_indexer(key)
        else:
            if key not in self.tsdframe.columns:
                raise IndexError(str(key))
            index = self.tsdframe.columns.get_indexer([key])

        if len(index) == 1:
            return self.tsdframe.__getitem__((slice(None, None, None), index[0]))
        else:
            return self.tsdframe.__getitem__(
                (slice(None, None, None), index), columns=key
            )


class _IntervalSetSliceHelper:
    """
    This class helps `IntervalSet` behaves like pandas.DataFrame for the `loc` function.

    Attributes
    ----------
    intervalset : `IntervalSet` to slice

    """

    def __init__(self, intervalset):
        """Class for `loc` slicing function

        Parameters
        ----------
        intervalset : IntervalSet

        """
        self.intervalset = intervalset

    def __getitem__(self, key):
        """Getters for `IntervalSet.loc`. Mimics pandas.DataFrame.

        Parameters
        ----------
        key : int, list or tuple

        Returns
        -------
        IntervalSet or Number or numpy.ndarray

        Raises
        ------
        IndexError

        """
        # Pickle backward compatibility
        try:
            metadata_columns = self.intervalset.metadata_columns
        except Exception:
            metadata_columns = []
        if key in ["start", "end"] + metadata_columns:
            return self.intervalset[key]
        elif isinstance(key, list):
            return self.intervalset[key]
        elif isinstance(key, int):
            return self.intervalset.values[key]
        else:
            if isinstance(key, tuple):
                if len(key) == 2:
                    if key[1] not in ["start", "end"] + metadata_columns:
                        raise IndexError
                    out = self.intervalset[key[0]][key[1]]
                    if len(out) == 1:
                        return out[0]
                    else:
                        return out
                else:
                    raise IndexError
            else:
                raise IndexError


def check_filename(filename):
    """Check if the filename is valid and return the path

    Parameters
    ----------
    filename : str or Path
        The filename

    Returns
    -------
    Path
        The path to the file

    Raises
    ------
    RuntimeError
        If the filename is a directory or the parent does not exist
    """
    filename = Path(filename).resolve()

    if filename.is_dir():
        raise RuntimeError("Invalid filename input. {} is directory.".format(filename))

    filename = filename.with_suffix(".npz")

    parent_folder = filename.parent
    if not parent_folder.exists():
        raise RuntimeError("Path {} does not exist.".format(parent_folder))

    return filename


def _convert_iter_to_str(array):
    """
    This function converts an array of arrays to array of strings.
    This help avoids a DeprecationWarning from numpy when printing an object with metadata
    """
    array = np.array(array)
    if array.ndim > 1:
        shape = array.shape
        array_str = np.empty(shape[0], dtype=object)
        # array = array.flatten()
        for i, arr in enumerate(array):
            if isinstance(arr, np.ndarray):
                array_str[i] = np.array2string(arr, precision=2)
        return array_str
    else:
        if np.issubdtype(array.dtype, np.floating):
            return np.around(array, decimals=2).astype(str)
        elif array.dtype == np.dtype("O"):
            return np.array(
                [
                    "".join(a.astype(str)) if hasattr(a, "astype") else str(a)
                    for a in array
                ]
            )
        else:
            return array.astype(str)


def add_docstring(method_name, cls):
    """Prepend super-class docstrings."""
    attr = getattr(cls, method_name, None)
    if attr is None:
        raise AttributeError(f"{cls.__name__} has no attribute {method_name}!")
    doc = attr.__doc__

    # Decorator to add the docstring
    def wrapper(func):
        func.__doc__ = "\n".join([doc, func.__doc__])  # Combine docstrings
        return func

    return wrapper


def _arg_as_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))


def modifies_time_axis(func, new_args, kwargs):
    """
    Return True if calling func(*new_args, **kwargs) would modify/move axis 0.
    Uses inspect.signature(bind_partial + apply_defaults) to get effective args.
    Conservative: if we can't determine array ndim, assume it *may* modify axis 0.
    """
    if func is np.flipud:
        return True
    if func is np.squeeze:
        return False  # This one should be handled by _initialize_tsd_output

    if func in (np.unwrap, np.copy, np.cumsum, np.nancumsum, np.cumprod, np.nancumprod):
        return False

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    bound = sig.bind_partial(*new_args, **kwargs)
    bound.apply_defaults()

    # Helper to get first array-like from positional args (conservative)
    arr = None
    if new_args:
        arr = new_args[0]
    else:
        # try common kw names
        for name in ("a", "arr", "array", "x", "m"):
            if name in bound.arguments:
                arr = bound.arguments[name]
                break

    ndim = getattr(arr, "ndim", None)
    if ndim is None:
        return False

    ### 1) single-axis arguments ###
    axis = bound.arguments.get("axis", inspect._empty)
    if axis is not inspect._empty:
        # axis=None usually means "all axes" for reductions => affects axis 0
        if (axis is None) or (axis == 0):
            return True
        if isinstance(axis, tuple) and (0 in axis):
            return True
        # axis might be negative; normalize if ndim known
        if axis < 0:
            normalized_axis = axis + ndim
            if func is np.expand_dims:
                if normalized_axis == -1:
                    # normalized_axis will be -1 when expanding first dimension
                    # normalized_axis = 0 will expand in the second dimension
                    return True
            else:
                if normalized_axis == 0:
                    return True

    # Special case for np.rollaxis
    if func is np.rollaxis:
        if bound.arguments.get("start", 0) == 0:
            return True
    # special case for np.rot90
    if func is np.rot90:
        if 0 in bound.arguments.get("axes", (0, 1)):
            return True

    ### 2) multi-axis permutation (e.g., transpose) ###
    axes = bound.arguments.get("axes", inspect._empty)
    if axes is not inspect._empty:
        if axes is None:
            return True  # all axes permuted => affects axis 0
        if _arg_as_sequence(axes):
            # if axis 0 is not at position 0 after permutation, it's moved
            idx = list(axes).index(0)
            # idx is new position of original axis 0
            if idx != 0:
                return True

    ### 3) moveaxis: source/destination can be ints or sequences ###
    for name in ("source", "destination"):
        val = bound.arguments.get(name, inspect._empty)
        if val is not inspect._empty:
            if val is None:
                continue
            elif (_arg_as_sequence(val)) and (0 in val):
                return True
            elif val == 0:
                return True

    ### 4) swapaxes / similar ###
    axis1 = bound.arguments.get("axis1", inspect._empty)
    axis2 = bound.arguments.get("axis2", inspect._empty)
    if (axis1 is not inspect._empty) and (axis1 == 0):
        return True
    if (axis2 is not inspect._empty) and (axis2 == 0):
        return True

    # If none of the checks triggered, assume axis 0 is not modified.
    return False
