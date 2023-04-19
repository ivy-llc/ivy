import ivy

# create an Ivy array
x = ivy.array([1, 2, 3, 4])

# compute the FFT using Ivy
y = ivy_fft(x)

# compute the IFFT using Ivy
z = ivy_ifft(y)

# print the results
print("Input: ", x)
print("FFT: ", y)
print("IFFT: ", z)
