import ivy

ivy.set_backend("paddle")

arr = (ivy.array([]), ivy.array([]))

print(arr)

arr = ivy.astype(arr, ivy.as_native_dtype("int64"))

print(arr)
