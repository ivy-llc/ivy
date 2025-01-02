
# Autograph_Ivy: Refactoring Autorgraph's source code to integrate with Ivy

- **Converters**: This module contains the transformation passes that are applied to the AST to transform it.
- **Core**: This module is the entry point of the code. The main function of interest here is the `to_functional_form()` function in the `api.py` file which takes as input a function(with SCF) and returns a transformed function(with FCF)  
- **Pyct**: This module contains the core logic for doing source code transformations. The good thing here is that it has been designed to serve as an independent module and hence contains no tensorflow-specific code. The module provides utilities for parsing, annotating and modfiying ASTs.

# Example 1 Simple While Loop
```
from control_flow_experimental.autograph_ivy.core.api import to_functional_form
import inspect

def while_with_if(i, x,w,b):
  while i <= 10:
    if i == 7:
      i += 1
    res = x*w+b
    i += 1
  return res

converted_fn = to_functional_form(while_with_if)
print(inspect.getsource(converted_fn))
```

returns the following transformed function: 
```
def ivy__while_with_if(i, x, w, b):

            def get_state_1():
                return (res, i)

            def set_state_1(vars_):
                nonlocal res, i
                (res, i) = vars_

            def loop_body():
                nonlocal res, i

                def get_state():
                    return (i,)

                def set_state(vars_):
                    nonlocal i
                    (i,) = vars_

                def if_body():
                    nonlocal i
                    i += 1

                def else_body():
                    nonlocal i
                    pass
                ivy.if_stmt((i == 7), if_body, else_body, get_state, set_state)
                res = ((x * w) + b)
                i += 1

            def loop_test():
                return (i <= 10)
            res = None
            ivy.while_stmt(loop_test, loop_body, get_state_1, set_state_1)
            return res

```

# Example 2: Handling of Call Trees/ Function Trees
```
def foo(x,y):
  if x > y:
    x+=20
  else:
    x+=10
  return x

def func(x,y):
    val = foo(x,y)
    i=0
    while i < val:
        y+=1
        z = ivy.array([1,2,3])
        w = tf.convert_to_tensor([1,2,3])
        i+=1

    return y
```
which gives the following output: 
```
def ivy__func(x, y):
            val = converted_call(foo, (x, y), None)
            i = 0

            def get_state():
                return (y, i)

            def set_state(vars_):
                nonlocal y, i
                (y, i) = vars_

            def loop_body():
                nonlocal y, i
                y += 1
                z = converted_call(ivy.array, ([1, 2, 3],), None)
                w = converted_call(tf.convert_to_tensor, ([1, 2, 3],), None)
                i += 1

            def loop_test():
                return (i < val)
            w = None
            z = None
            ivy.while_stmt(loop_test, loop_body, get_state, set_state)
            return y
```
Here the function `foo` is not converted directly. Rather, all `Call()` nodes in the AST are replaced with a call to the `converted_call` function. When executing the (transformed) function, the `converted_call` function will dynamically decide whether to recurse into the input function or whether to execute it as it is (based on some user defined criteria). Thus for the above example, the calls to `ivy.array([1,2,3], None)` and `tf.convert_to_tensor, ([1, 2, 3], None)` would get executed as is, whereas the call to `foo, (x, y), None)` would get recursively transformed into its FCF form.)


Currently, the `Converters` module contains passes that support the following: 
- If-else
- While loops 
- Breaks/Continues
- List Comprehension
- Call Trees and Closures