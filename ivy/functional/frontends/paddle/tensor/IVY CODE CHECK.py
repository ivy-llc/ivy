# local
import ivy

def greater_equal(x, y):
    return ivy.greater_equal(x, y)

# Example usage
# Sample inputs to demonstrate the expected behavior
# Result returns the truth value of x>=y
x = ivy.array([1, 2, 3])
y = ivy.array([1, 3, 2])
result = ivy.greater_equal(x, y)
print(result)
