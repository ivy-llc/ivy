#!/bin/bash
# Sensure our script doesn't fail silently
set -efu

# Some basic parameters that should bery rarely change
IVY_PREFIX="/ivy"
DOCKER_IMAGE="unifyai/ivy:latest"
PYTEST_CMD=("python3" "-m" "pytest")

if [[ "-h" == "$1" || "--help" == "$1" ]]; then
    echo "USAGE: $0 [TESTS..] -- [PYTEST OPTIONS...]"
    echo "To run all tests pass no options or use -- as the first one"
    echo "  $0"
    echo "  $0 --"
    echo "To run all tests on a specific file, use its path as the first argument"
    echo "  $0 ivy_tests/test_ivy/test_functional/test_core/test_general.py"
    echo "To run a specific test on specific file, use its path and function name as the first argument"
    echo "  $0 ivy_tests/test_ivy/test_functional/test_core/test_general.py::test_array_equal"
    echo "To run multiple tests just combine them as you would intuitively guess"
    echo "  $0 ivy_tests/test_ivy/test_functional/test_core/test_general.py::test_array_equal ivy_tests/test_ivy/test_functional/test_core/test_random.py ivy_tests/test_ivy/test_functional/test_core/test_general.py::test_shape"
    echo "Some examples of pytest options"
    echo "  --no-header"
    echo "  --no-summary"
    echo "  -q same as --quiet"
    echo "Remember to always use relative paths from the repository root"
    exit 1
fi

RAW_ARGS=("$@") # These quotation marks are indispensible to avoid bugs with filenames with spaces
PYTEST_TESTS=()
PYTEST_ARGS=()
ARGS_SPLIT_INDEX=-1 # Holds the position of the '--' argument
for I in ${!RAW_ARGS[@]}; do
    if [[ "--" == "${RAW_ARGS[$I]}" ]]; then
        ARGS_SPLIT_INDEX="${I}"
        break
    fi
done
for I in ${!RAW_ARGS[@]}; do
    if (( $I < $ARGS_SPLIT_INDEX || $ARGS_SPLIT_INDEX < 0)); then
        PYTEST_TESTS+=("${IVY_PREFIX}/${RAW_ARGS[$I]}")
    elif (( $I > $ARGS_SPLIT_INDEX && $ARGS_SPLIT_INDEX > 0)); then
        PYTEST_ARGS+=("${RAW_ARGS[$I]}")
    fi
done

PWD="$(pwd)"
set -x
# These quotation marks are indispensible to avoid bugs with filenames with spaces
docker run --rm -it -v "${PWD}":"${IVY_PREFIX}" "${DOCKER_IMAGE}" "${PYTEST_CMD[@]}" "${PYTEST_TESTS[@]}" "${PYTEST_ARGS[@]}"
set +x

exit 0

set -x

if [[ -z "${REQUESTED_TESTS}" || "--" -eq "${REQUESTED_TESTS}" ]]; then
    docker run --rm -it -v "$(pwd)":/ivy unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/ ${@: 2}
else
    docker run --rm -it -v "$(pwd)":/ivy unifyai/ivy:latest python3 -m pytest "ivy/${REQUESTED_TESTS}" ${@: 2}
fi