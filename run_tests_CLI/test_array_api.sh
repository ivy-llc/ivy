#!/bin/bash -e
python3 ivy_tests/write_array_api_tests_k_flag.py
# shellcheck disable=SC2155
export ARRAY_API_TESTS_K_FLAG=$(cat ivy_tests/.array_api_tests_k_flag_$1)
if [ "$1" = "torch" ]; then
  ARRAY_API_TESTS_K_FLAG="${ARRAY_API_TESTS_K_FLAG} and not (uint16 or uint32 or uint64)"
fi
mkdir -p .hypothesis
# shellcheck disable=SC2046
docker run --rm --env IVY_BACKEND="$1" --env ARRAY_API_TESTS_MODULE="ivy" -v $(pwd):/ivy -v $(pwd)/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/test_array_api/array_api_tests/"$2".py -k "$ARRAY_API_TESTS_K_FLAG" -vv
