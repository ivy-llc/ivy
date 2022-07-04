#!/bin/bash -e
python3 ivy_tests/write_array_api_tests_k_flag.py
# shellcheck disable=SC2155
export ARRAY_API_TESTS_K_FLAG=$(cat ivy_tests/.array_api_tests_k_flag_$1)
if [ "$1" = "torch" ]; then
  ARRAY_API_TESTS_K_FLAG="${ARRAY_API_TESTS_K_FLAG} and not (uint16 or uint32 or uint64)"
fi
mkdir -p .hypothesis
# SG
cat << EOF >> ivy_tests/test_array_api/skips.txt
ivy/ivy_tests/test_array_api/array_api_tests/test_array_object.py::test_getitem_masking
# copy not implemented
array_api_tests/test_creation_functions.py::test_asarray_arrays
# https://github.com/numpy/numpy/issues/20870
array_api_tests/test_data_type_functions.py::test_can_cast
# The return dtype for trace is not consistent in the spec
# https://github.com/data-apis/array-api/issues/202#issuecomment-952529197
array_api_tests/test_linalg.py::test_trace
# waiting on NumPy to allow/revert distinct NaNs for np.unique
# https://github.com/numpy/numpy/issues/20326#issuecomment-1012380448
array_api_tests/test_set_functions.py
# https://github.com/numpy/numpy/issues/21373
array_api_tests/test_array_object.py::test_getitem
# missing copy arg
array_api_tests/test_signatures.py::test_func_signature[reshape]
# https://github.com/numpy/numpy/issues/21211
array_api_tests/test_special_cases.py::test_iop[__iadd__(x1_i is -0 and x2_i is -0) -> -0]
# https://github.com/numpy/numpy/issues/21213
array_api_tests/test_special_cases.py::test_iop[__ipow__(x1_i is -infinity and x2_i > 0 and not (x2_i.is_integer() and x2_i % 2 == 1)) -> +infinity]
array_api_tests/test_special_cases.py::test_iop[__ipow__(x1_i is -0 and x2_i > 0 and not (x2_i.is_integer() and x2_i % 2 == 1)) -> +0]
# noted diversions from spec
array_api_tests/test_special_cases.py::test_binary[floor_divide(x1_i is +infinity and isfinite(x2_i) and x2_i > 0) -> +infinity]
array_api_tests/test_special_cases.py::test_binary[floor_divide(x1_i is +infinity and isfinite(x2_i) and x2_i < 0) -> -infinity]
array_api_tests/test_special_cases.py::test_binary[floor_divide(x1_i is -infinity and isfinite(x2_i) and x2_i > 0) -> -infinity]
array_api_tests/test_special_cases.py::test_binary[floor_divide(x1_i is -infinity and isfinite(x2_i) and x2_i < 0) -> +infinity]
array_api_tests/test_special_cases.py::test_binary[floor_divide(isfinite(x1_i) and x1_i > 0 and x2_i is -infinity) -> -0]
array_api_tests/test_special_cases.py::test_binary[floor_divide(isfinite(x1_i) and x1_i < 0 and x2_i is +infinity) -> -0]
array_api_tests/test_special_cases.py::test_binary[__floordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i > 0) -> +infinity]
array_api_tests/test_special_cases.py::test_binary[__floordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i < 0) -> -infinity]
array_api_tests/test_special_cases.py::test_binary[__floordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i > 0) -> -infinity]
array_api_tests/test_special_cases.py::test_binary[__floordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i < 0) -> +infinity]
array_api_tests/test_special_cases.py::test_binary[__floordiv__(isfinite(x1_i) and x1_i > 0 and x2_i is -infinity) -> -0]
array_api_tests/test_special_cases.py::test_binary[__floordiv__(isfinite(x1_i) and x1_i < 0 and x2_i is +infinity) -> -0]
array_api_tests/test_special_cases.py::test_iop[__ifloordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i > 0) -> +infinity]
array_api_tests/test_special_cases.py::test_iop[__ifloordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i < 0) -> -infinity]
array_api_tests/test_special_cases.py::test_iop[__ifloordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i > 0) -> -infinity]
array_api_tests/test_special_cases.py::test_iop[__ifloordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i < 0) -> +infinity]
array_api_tests/test_special_cases.py::test_iop[__ifloordiv__(isfinite(x1_i) and x1_i > 0 and x2_i is -infinity) -> -0]
array_api_tests/test_special_cases.py::test_iop[__ifloordiv__(isfinite(x1_i) and x1_i < 0 and x2_i is +infinity) -> -0]
EOF
# shellcheck disable=SC2046
docker run --rm --env IVY_BACKEND="$1" --env ARRAY_API_TESTS_MODULE="ivy" -v $(pwd):/ivy -v $(pwd)/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/test_array_api -k "$ARRAY_API_TESTS_K_FLAG" -vv
