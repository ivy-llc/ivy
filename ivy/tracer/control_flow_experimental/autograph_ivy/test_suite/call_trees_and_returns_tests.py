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


def is_prime(x):
    if x < 2:
        return False
    i = 2
    while i <= x // 2:
        if x % i == 0:
            return False
        i += 1
    return True


def find_largest_prime_number():
    numbers = ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    largest_prime = ivy.array(0)
    i = 0
    while i < len(numbers):
        if is_prime(numbers[i]):
            if numbers[i] > largest_prime:
                largest_prime = numbers[i]
        i += 1
    return largest_prime


def find_prime_factors(x):
    prime_factors = ivy.array([])
    i = 2
    while x > 1:
        if x % i == 0:
            prime_factors = ivy.concat((prime_factors, ivy.array([i])))
            x /= i
        else:
            i += 1
    return prime_factors


def find_largest_prime_factor():
    numbers = ivy.array([13, 25, 19, 55, 81, 105, 37])
    largest_prime_factor = 0
    i = 0
    while i < len(numbers):
        prime_factors = find_prime_factors(numbers[i])
        if len(prime_factors) > 0:
            if prime_factors[-1] > largest_prime_factor:
                largest_prime_factor = prime_factors[-1]
        i += 1
    return largest_prime_factor


def find_even_odd_numbers(nums):
    numbers = ivy.array(nums)
    even_numbers = ivy.array([])
    odd_numbers = ivy.array([])
    i = 0
    while i < len(numbers):
        if numbers[i] % 2 == 0:
            even_numbers = ivy.concat((even_numbers, ivy.array([numbers[i]])))
        else:
            odd_numbers = ivy.concat((odd_numbers, ivy.array([numbers[i]])))
        i += 1
    if len(numbers) % 2 == 0:
        return even_numbers
    else:
        return odd_numbers


# Test the function

# even_len_list = ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# test_function(find_even_odd_numbers, even_len_list)
odd_len_list = ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
test_function(find_even_odd_numbers, odd_len_list)

# Test the function (uncomment any one)

# test_function(find_largest_prime_number)
# test_function(find_largest_prime_factor)
# test_function(find_even_odd_numbers, ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# test_function(find_even_odd_numbers, ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
