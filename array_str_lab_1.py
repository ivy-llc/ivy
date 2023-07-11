import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)

ivy.set_backend("numpy")


@to_ivy_arrays_and_back
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    try:
        print("===============================================")
        print("a= ", a)
        print("max_line_width= ", max_line_width)
        print("precision= ", precision)
        print("suppress_small= ", suppress_small)
        print("-----------------------------------------------")
        a = ivy.reshape(a, (-1,))
        print(f"reshaped a: {a}")
        if precision is not None and ivy.dtype(a)!="bool":
            arr_str = ivy.to_numpy(ivy.round(a, decimals=precision)).__str__()

        arr_str = ivy.to_numpy(a).__str__()
        arr_str = arr_str.replace('inf', 'infj').replace('nan', 'nanj')

        if suppress_small is not None:
            arr_str_lines = arr_str.splitlines()
            for i, line in enumerate(arr_str_lines):
                if suppress_small and line.startswith(' ' * 5 + '.'):
                    arr_str_lines[i] = ' ' * 5 + '0'
            arr_str = '\n'.join(arr_str_lines)

        def break_lines(text, max_line_width):
            broken_lines = []
            line_start = 0
            line_end = max_line_width
            while line_start < len(text):
                broken_lines.append(text[line_start:line_end])
                line_start = line_end
                line_end += max_line_width
            return '\n'.join(broken_lines)

        if max_line_width is not None:
            arr_str = break_lines(arr_str, max_line_width)

        return arr_str
    except Exception as e:
        print("All Mighty Error: ", e)


a=  ivy.array(0)
max_line_width=  1
precision=  1
suppress_small=  False
print(array_str(a, max_line_width, precision, suppress_small))


# from ground truth backend:  [array('[False]', dtype='<U7')]
# from ivy backend:  []