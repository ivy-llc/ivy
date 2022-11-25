from distutils.core import setup, Extension
 
def main():
   setup(name="Cython_wrap",
         version="1.0.0",
         description="C interface for func_wrapping in ivy",
         ext_modules=[Extension("Cython_func_wrapper", ["Cython_func_wrapper.c"])])
 
if __name__ == "__main__":
   main()
