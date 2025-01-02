from control_flow_experimental.autograph_ivy.core.api import to_functional_form
import ivy


IVY_BACKENDS = ["numpy", "torch", "jax", "tensorflow", "paddle"]


def test_function(func, *args, backends=IVY_BACKENDS, **kwargs):
    for backend in backends:
        # Set the backend
        ivy.set_backend(backend)

        # Run the function and get the output
        output = func(*args, **kwargs)
        print(f"Original function output ({backend} backend): {output}")

        # Convert the function to its functional form
        converted_func = to_functional_form(func)

        # Run the converted function and get its output
        converted_output = converted_func(*args, **kwargs)
        print(f"Converted function output ({backend} backend): {converted_output}")

        # Perform the assertion that the outputs are the same
        assert ivy.array_equal(
            output, converted_output
        ), f"Outputs do not match for function {func.__name__} and backend {backend}"


def simple_while_loop():
    result = ivy.array(0)
    i = ivy.array(0)
    while i < 10:
        result += i
        i += 1
    return result


def while_with_if():
    result = ivy.array(0)
    i = ivy.array(0)
    while i < 10:
        if i % 2 == 0:
            result += i
        else:
            result -= i
        i += 1
    return result


def nested_while_with_if():
    result = ivy.array(0)
    i = ivy.array(0)
    while i < 10:
        if i % 2 == 0:
            j = ivy.array(0)
            while j < i:
                result += j
                j += 1
        else:
            result -= i
        i += 1
    return result


# Test the function (uncomment any one)

# test_function(while_with_break)
# test_function(while_with_if)
# test_function(simple_while_loop)
# test_function(nested_while_with_if)
