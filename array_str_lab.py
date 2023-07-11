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
        # handles if precision is none or if invalid
        ivy.set_array_significant_figures(precision)
        print(f"Done: 1")

        a = ivy.reshape(a, (-1,))

        def round_suppress(ele, precision_):
            count_after_decimal = str(ele)[::-1].find(".")
            if suppress_small and (count_after_decimal > precision_):
                # supress to zero
                print(f"Done: 2")
                return 0
            else:
                print(f"Done: 2")
                return round(ele, precision_)

        print(f"Optional: 2 | a.shape: {a.shape}")
        # if len(a.shape) != 0:
        if a.size != 0:
            # ivy.map(fn=round_suppress, constant=ivy.array_significant_figures_stack[-1], unique = ivy.flatten(a))
            a = list(
                map(
                    round_suppress,
                    ivy.to_list(a),
                    [ivy.array_significant_figures_stack[-1]],
                )
            )
        a = str(a)
        if max_line_width is None:
            # TODO: max_line_width handler func
            max_line_width = 75
        print(f"Done: 3")

        def chop_string(string, chunk_size):
            chopped_string = ""
            for i in range(0, len(string), chunk_size):
                chopped_string += string[i : i + chunk_size] + "\n"
            print(f"Done: 4")
            return chopped_string

        a = chop_string(a, max_line_width)
        print(f"Done: 5")
        return a
    except Exception as e:
        print("All Mighty Error: ", e)


a = ivy.array(0)
max_line_width = 1
precision = 1
suppress_small = False
print(array_str(a, max_line_width, precision, suppress_small))
