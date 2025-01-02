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


def find_sum_subarray():
    num_list = ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    target_sum = ivy.array(15)

    current_sum = ivy.array(0)
    start_index = 0
    end_index = 0

    while end_index < len(num_list):
        if current_sum < target_sum:
            current_sum += num_list[end_index]
            end_index += 1

        elif current_sum == target_sum:
            break

        else:
            current_sum -= num_list[start_index]
            start_index += 1

    return num_list[start_index:end_index]


def find_even_numbers():
    numbers = ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    even_numbers = ivy.array([])
    i = 0
    while i < len(numbers):
        if numbers[i] % 2 == 0:
            even_numbers = ivy.concat((even_numbers, ivy.array([numbers[i]])))
        else:
            i += 1
            continue
        i += 1
    return even_numbers


def find_largest_prime_number():
    numbers = ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    largest_prime = 0
    i = 0
    while i < len(numbers):
        is_prime = True
        j = 2
        while j < numbers[i]:
            if numbers[i] % j == 0:
                is_prime = False
                break
            j += 1
        if not is_prime:
            i += 1
            continue
        if numbers[i] > largest_prime:
            largest_prime = numbers[i]
        i += 1
    return largest_prime


# Test the function (uncomment any one)

# test_function(find_even_numbers)
# test_function(find_sum_subarray)
# test_function(find_largest_prime_number)
