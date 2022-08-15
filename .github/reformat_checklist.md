
#### IMPORTANT NOTICE 🚨:
The ivy docs represent the ground truth for the task descriptions and this checklist should only be used as a supplementary item to aid with the review process.

#### LEGEND 🗺:
❌ :  Check item is not completed. 
✅ :  Check item is ready for review.
🆘 :  Stuck/Doubting implementation (PR author should add comments explaining why).
⏩ :  Check is not applicable to function (skip).
🆗 :  Check item is already implemented and does not require any edits.

#### CHECKS 📑:
1. - [ ] ❌:  Remove all lambda and direct bindings for the backend functions in [ivy/functional/backends](https://github.com/unifyai/ivy/tree/master/ivy/functional/backends).
2. - [ ] ❌: Implement the following if they don't exist: 
       1. - [ ]  ❌: The `ivy.Array` instance method in [ivy/array/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/array/{{ .category_name }}.py).
       2. - [ ]  ❌: The `ivy.Array` special method in [ivy/array/array.py](https://github.com/unifyai/ivy/blob/master/ivy/array/array.py).
       3. - [ ]  ❌: The `ivy.Array` reverse special method in [ivy/array/array.py](https://github.com/unifyai/ivy/blob/master/ivy/array/array.py).
       4. - [ ] ❌: The `ivy.Container` static method in [ivy/container/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/container/{{ .category_name }}.py).
       5. - [ ] ❌: The `ivy.Container` instance method in [ivy/container/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/container/{{ .category_name }}.py).
       6. - [ ]  ❌:  The `ivy.Container` special method in [ivy/container/container.py](https://github.com/unifyai/ivy/blob/master/ivy/container/container.py).
       7. - [ ]  ❌: The `ivy.Container` reverse special method in [ivy/container/container.py](https://github.com/unifyai/ivy/blob/master/ivy/container/container.py).
3. - [ ] ❌:  Make sure that the aforementioned methods are added into the correct category-specific parent class, such as  `ivy.ArrayWithElementwise`,  `ivy.ContainerWithManipulation`  etc.
4. - [ ] ❌:  Correct all of the  [Function Arguments and the type hints](https://lets-unify.ai/ivy/deep_dive/10_function_arguments.html#function-arguments) for every function  **and**  its  _relevant methods_, including those you did not implement yourself. 
5. - [ ] ❌: Add the correct  [Docstrings](https://lets-unify.ai/ivy/deep_dive/12_docstrings.html#docstrings)  to every function  **and**  its  _relevant methods_, including those you did not implement yourself. The following should be added: 
       1. - [ ] ❌:   <a name="ref1"></a> The function's [Array API standard](https://data-apis.org/array-api/latest/index.html) description in [ivy/functional/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/functional/ivy/{{ .category_name }}.py). If the function is not part of the Array API standard then a description of similar style should be added to the same file. 
	The following modifications should be made to the description:
              - [ ] ❌:  Remove type definitions in the `Parameters` and `Returns` sections. 
              - [ ] ❌:  Add `out` to the `Parameters` section if function accepts an `out` argument.
              - [ ] ❌:  Replace `out` with `ret` in the `Returns` section.
       2. - [ ] ❌:  Reference to docstring for ivy.function_name ([5.a](#ref1)) for the function description **and** modified `Parameters` and `Returns` sections as described in [the docs](https://lets-unify.ai/ivy/deep_dive/12_docstrings.html#docstrings) in:
              - [ ] ❌:  [ivy/array/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/array/{{ .category_name }}.py).
              - [ ] ❌:  [ivy/container/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/container/{{ .category_name }}.py).
              - [ ] ❌:   [ivy/array/array.py](https://github.com/unifyai/ivy/blob/master/ivy/array/array.py) if the function has a special method  ( like `__function_name__` ).
              - [ ] ❌:  [ivy/array/array.py](https://github.com/unifyai/ivy/blob/master/ivy/array/array.py) if the function has a reverse special method  ( like `__function_name__` ).
              - [ ] ❌: [ivy/container/container.py](https://github.com/unifyai/ivy/blob/master/ivy/container/container.py) if the function has a special method ( like `__function_name__` ).
              - [ ] ❌:  [ivy/container/container.py](https://github.com/unifyai/ivy/blob/master/ivy/container/container.py) if the function has a reverse special method  ( like `__function_name__` ).
6. - [ ] ❌: Add thorough  [Docstring Examples](https://lets-unify.ai/ivy/deep_dive/13_docstring_examples.html#docstring-examples)  for every function  **and**  its  _relevant methods_  and ensure they pass the docstring tests.
	
		**Functional Examples** in [ivy/functional/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/functional/ivy/{{ .category_name }}.py).

		1. - [ ] ❌: Cover all possible variants for each of the arguments independently (not combinatorily).  
	 	2. - [ ] ❌: Vary the values and input shapes considerably between examples.
	 	3. - [ ] ❌: Start out simple and get more complex with each example.
	 	4. - [ ] ❌: Show an example with:  
			   - [ ] ❌: `out` unused.  
			   - [ ] ❌: `out` used to update a new array y.
			   - [ ] ❌: `out` used to inplace update the input array x (if x has the same dtype and shape as the return). 
	 	5. - [ ] ❌: If broadcasting is relevant for the function, then show examples which highlight this. 
		
		**Nestable Function Examples** in [ivy/functional/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/functional/ivy/{{ .category_name }}.py).
		Only if the function supports nestable operations.
	 	
	 	6. - [ ] ❌: <a name="ref2"></a> Add an example that passes in an  `ivy.Container`  instance in place of one of the arguments. 
	 	7. - [ ] ❌: <a name="ref3"></a> Add an example passes in  `ivy.Container`  instances for multiple arguments.
		
		**Container Static Method Examples** in [ivy/container/{{ .category_name }}e.py](https://github.com/unifyai/ivy/blob/master/ivy/container/{{ .category_name }}.py).

	 	8. - [ ] ❌: The example from point ([6.f](#ref2)) should be replicated, but added to the  `ivy.Container`  **static method** docstring in with  `ivy.<func_name>`  replaced with  `ivy.Container.static_<func_name>`  in the example.
	 	9. - [ ] ❌: The example from point ([6.g](#ref3)) should be replicated, but added to the  `ivy.Container`  **static method** docstring, with  `ivy.<func_name>`  replaced with  `ivy.Container.static_<func_name>`  in the example.
	
		**Array Instance Method Example** in [ivy/array/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/array/{{ .category_name }}).
	 	
		10. - [ ] ❌: Call this instance method of the  `ivy.Array`  class.
	
		**Container Instance Method Example** in [ivy/container/{{ .category_name }}.py](https://github.com/unifyai/ivy/blob/master/ivy/container/{{ .category_name }}.py).
	 	
		11. - [ ] ❌: Call this instance method of the  `ivy.Container`  class.
		 
		**Array Operator Examples** in [ivy/array/array.py](https://github.com/unifyai/ivy/blob/master/ivy/array/array.py).
	 	
		12. - [ ] ❌: Call the operator on two  `ivy.Array`  instances.
	 	13. - [ ] ❌: Call the operator with an  `ivy.Array`  instance on the left and  `ivy.Container`  on the right.
	
		**Array Reverse Operator Example** in [ivy/array/array.py](https://github.com/unifyai/ivy/blob/master/ivy/array/array.py).
	 	
		14.  - [ ] ❌: Call the operator with a  `Number`  on the left and an  `ivy.Array`  instance on the right.
	
		**Container Operator Examples** in [ivy/container/container.py](https://github.com/unifyai/ivy/blob/master/ivy/container/container.py).
	 	
		15. - [ ] ❌: Call the operator on two `ivy.Container` instances containing Number instances at the leaves.
	 	16. - [ ] ❌: Call the operator on two `ivy.Container` instances containing `ivy.Array` instances at the leaves.
	 	17. - [ ] ❌: Call the operator with an `ivy.Container` instance on the left and `ivy.Array` on the right.

		**Container Reverse Operator Example** in [ivy/container/container.py](https://github.com/unifyai/ivy/blob/master/ivy/container/container.py).

		18. - [ ] ❌: Following example in the [`ivy.Container.__radd__`](https://github.com/unifyai/ivy/blob/e28a3cfd8a4527066d0d92d48a9e849c9f367a39/ivy/container/container.py#L173) docstring, with the operator called with a `Number` on the left and an `ivy.Container` instance on the right.

		**Tests**

		19. - [ ] ❌: Docstring examples tests passing.
		20. - [ ] ❌: Lint checks passing.
