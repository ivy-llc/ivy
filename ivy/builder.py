import os

# give your current compiler flag
cc = "gcc"

# Cythonize ivy's function wrapping with embeded mode
os.system("cython Cython_func_wrapper.py --embed")

# Build the C extention module using gcc or clang
os.system(f"${cc} python3 setup.py install")
