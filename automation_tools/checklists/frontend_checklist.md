
## Frontend Task Checklist
#### IMPORTANT NOTICE 🚨:
The [Ivy Docs](https://unify.ai/docs/ivy/) represent the ground truth for the task descriptions and this checklist should only be used as a supplementary item to aid with the review process.

Please note that the contributor is not expected to understand everything in the checklist. It's mainly here for the reviewer to make sure everything has been done correctly 🙂

#### LEGEND 🗺:
- ❌ :  Check item is not completed. 
- ✅ :  Check item is ready for review.
- 🆘 :  Stuck/Doubting implementation (PR author should add comments explaining why).
- ⏩ :  Check is not applicable to function (skip).
- 🆗 :  Check item is implemented and does not require any edits.

#### CHECKS 📑:
1. - [ ] ❌: The function/method definition is not missing any of the original arguments.
2. - [ ] ❌: In case the function/method to be implemented is an alias of an existing function/method:
       1. - [ ] ❌: It is being declared as such by setting `fun1 = fun2`, rather than being re-implemented from scratch.
       2. - [ ] ❌: The alias is added to the existing function/method's test in the `aliases` parameter of `handle_frontend_test`/`handle_frontend_method`.
3. - [ ] ❌: The naming of the function/method and its arguments exactly matches the original.
4. - [ ] ❌: No defined argument is being ignored in the function/method's implementation.
5. - [ ] ❌: In special cases where an argument's implementation should be pending due to an incomplete superset of an ivy function:
       1. - [ ] ❌: A descriptive comment has been left under the `Implement superset behavior` ToDo list in https://github.com/unifyai/ivy/issues/6406.
       2. - [ ] ❌: A ToDo comment has been added prompting to pass the frontend argument to the ivy function whose behavior is to be extended.
6. - [ ] ❌: In case a frontend function is being added:
       1. - [ ] ❌: It is a composition of ivy functions.
       2. - [ ] ❌: In case the needed composition is long (using numerous ivy functions), a `Missing Function Suggestion` issue has been opened to suggest a new ivy function should be added to shorten the frontend implementation. 
       3. - [ ] ❌: `@to_ivy_arrays_and_back` has been added to the function.
7. - [ ] ❌: In case a frontend method is being added:
       1. - [ ] ❌: It is composed of existing frontend functions or methods. 
       2. - [ ] ❌: If a required frontend function has not yet been added, the method may be implemented as a composition of ivy functions, making sure that:
              - [ ] ❌: `@to_ivy_arrays_and_back` has been added to the method.
              - [ ] ❌: A ToDo comment has been made prompting to remove the decorator and update the implementation as soon as the missing function has been added.
8. - [ ] ❌: The function/method's test has been added (except in the alias case mentioned in <2>):
       1. - [ ] ❌: All supported arguments are being generated in `handle_frontend_test`/`handle_frontend_method` and passed to `test_frontend_function`/`test_frontend_method`. 
       2. - [ ] ❌: The argument generation covers all possible supported values. Array sizes, dimensions, and axes adhere to the full supported set of the original function/method.
       3. - [ ] ❌: The `available_dtypes` parameter passed to the helper generating the function/method's input array is set to `helpers.get_dtypes("valid")`. If there are unsupported dtypes that cause the test to fail, they should be handled by adding `@with_supported_dtypes`/`@with_unsupported_dtype` to the function/method.
9. - [ ] ❌: The PR is not introducing any test failures.
       1. - [ ] ❌: The lint checks are passing.
       2. - [ ] ❌: The implemented test is passing for all backends.
10. - [ ] ❌: The PR `closes` a `Sub Task` issue linked to one of the open frontend ToDo lists.
11. - [ ] ❌: The function/method and its test have been added to the correct `.py` files corresponding to the addressed ToDo list.
12. - [ ] ❌: The PR only contains changes relevant to the addressed subtask.