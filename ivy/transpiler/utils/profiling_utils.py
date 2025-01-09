import time
import functools


class Timings:
    def __init__(self):
        self.dict = {}

    def update(self, name, timing):
        if name in self.dict:
            self.dict[name][0] += timing
            self.dict[name][1] += 1
        else:
            self.dict[name] = [timing, 1]

    def __repr__(self):
        return str(self.dict)


timings = Timings()

# names and call counts of translated functions.
call_counts = {}

# translated function objects
translated_objects = {}

# args and kwargs to use for profiling functions.
args_kwargs = {}

# map from function names post-translation to the function objects they were translated from.
name_map = {}

# associate translated objects with native versions
translated_native_pairs = {}


def profiling_timing_decorator(fn):
    if fn.__wrapped__:
        return fn

    def decorated(*args, **kwargs):
        time_ = time.perf_counter()
        ret = fn(*args, **kwargs)
        time__ = time.perf_counter()
        timings.update(fn.__name__ + "_" + str(id(fn))[:5], time__ - time_)
        return ret

    return decorated


def profiling_logging_decorator(fn):
    if fn.__name__ == "call":
        return fn
    if hasattr(fn, "__wrapped__"):
        return fn

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        if _name(fn) in call_counts:
            call_counts[_name(fn)] += 1
            args_kwargs[_name(fn)] = (args, kwargs)
        else:
            call_counts[_name(fn)] = 0
            args_kwargs[_name(fn)] = (args, kwargs)
        return fn(*args, **kwargs)

    if _name(fn) not in translated_objects:
        translated_objects[_name(fn)] = fn
    return decorated


def _name(fn):
    return fn.__name__ + "_" + str(id(fn))[:5]


should_profile = {}


def get_function_map():
    map = {}
    for function_name in call_counts.keys():
        if not function_name[:-6] in name_map:
            continue
        map[function_name] = name_map[function_name[:-6]]
    return map


def _to_tf(args, kwargs):
    try:
        import tensorflow as tf
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`tensorflow` and/or numpy not found installed on your system. Please proceed "
            "to install them and restart your interpreter to see the changes."
        ) from exc

    r_args = list()
    r_kwargs = {}
    for arg in args:
        if isinstance(arg, tf.Tensor):
            r_args.append(tf.constant(np.asarray(arg)))
        else:
            r_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, tf.Tensor):
            r_kwargs[k] = tf.constant(np.asarray(v))
        else:
            r_kwargs[k] = v
    return r_args, r_kwargs


def _to_torch(args, kwargs):
    try:
        import tensorflow as tf
        import torch
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`tensorflow`, `torch` and/or `numpy` not found installed on your system. Please proceed "
            "to install them and restart your interpreter to see the changes."
        ) from exc
    r_args = list()
    r_kwargs = {}
    for arg in args:
        if isinstance(arg, (tf.Tensor, tf.Variable)):
            r_args.append(torch.tensor(np.asarray(arg)))
        else:
            r_args.append(arg)
    for k, v in kwargs.items():

        if isinstance(v, (tf.Tensor, tf.Variable)):
            r_kwargs[k] = torch.tensor(np.asarray(v))
        else:
            r_kwargs[k] = v
    return r_args, r_kwargs


def time_all_functions(compiled=True):

    filter()

    results = {}
    failed_functions = []
    objects = translated_native_pairs
    for k, v in objects.items():
        translated = v[0]
        native = v[1]
        time1 = None
        time2 = None
        try:
            # todo: use framework compilers
            # use backend compiler here. for some reason, the default tf.function commented here is quite slow.
            compiled = translated  # tf.function(translated)

            args, kwargs = args_kwargs[k]

            args, kwargs = _to_tf(args_kwargs[k][0], args_kwargs[k][1])

            compiled.__call__(*args, **kwargs)

            time_ = time.perf_counter()

            for x in range(1000):
                compiled.__call__(*args, **kwargs)

            time__ = time.perf_counter()

            time1 = time__ - time_

        except Exception as e:
            failed_functions.append(k)

        try:
            args, kwargs = _to_torch(args_kwargs[k][0], args_kwargs[k][1])

            time_ = time.perf_counter()

            for x in range(1000):
                native.__call__(*args, **kwargs)

            time__ = time.perf_counter()

            time2 = time__ - time_

        except Exception as e:
            failed_functions.append(k)

        results[k] = (time1, time2)

    return results


def format_results(results):

    table = []
    width = 0

    for k, v in results.items():

        # Name, translated_time, native_time, slow_down (%), times_called

        if v[0] is None or v[1] is None:
            slow_down = "unknown"
        else:
            slow_down = (v[0] / v[1] - 1) * 100

        row = [k[:-6], v[0], v[1], slow_down, call_counts[k]]

        table.append(row)

    table.sort(reverse=True, key=lambda row: row[3] if isinstance(row[3], float) else 0)

    for i in range(len(table)):
        row = table[i]

        row[0] = row[0].ljust(15)

        if row[1] is not None:
            row[1] = "{:.4f}ms".format(row[1])

        if row[2] is not None:
            row[2] = "{:.4f}ms".format(row[2])

        if row[3] != "unknown":
            row[3] = "{:.2f}%".format(row[3])

        if row[4] is not None:
            row[4] = str(row[4])

        width = max(width, len(row[0]))

        table[i] = row

    table.insert(0, ["Name", "Translated", "Native", "Slowdown", "Times called"])

    out = ""

    for row in table:
        row[0] = str(row[0]).ljust(width)
        row[1] = str(row[1]).ljust(10)
        row[2] = str(row[2]).ljust(10)
        row[3] = str(row[3]).ljust(10)
        out += row[0] + "|" + row[1] + "|" + row[2] + "|" + row[3] + "|" + row[4] + "\n"

    return out


def filter():
    """
    Extract translated functions that correspond to original framework functions.
    """
    filtered = {}
    for function_name in call_counts.keys():
        if not function_name[:-6] in name_map:
            continue
        function = name_map[function_name[:-6]]
        filtered[function_name] = function
        translated_native_pairs[function_name] = (
            translated_objects[function_name],
            function,
        )

        while True:
            break
            if not function.__name__ in name_map:
                filtered[function_name] = function
                translated_native_pairs[function_name] = (
                    translated_objects[function_name],
                    function,
                )
                break
            if "Translated" in function.__name__:
                filtered[function_name] = name_map[function.__name__]
                translated_native_pairs[function_name] = (
                    translated_objects[function_name],
                    name_map[function.__name__],
                )
                break
            function = name_map[function.__name__]

    return filtered


def show():
    print(format_results(time_all_functions()))
