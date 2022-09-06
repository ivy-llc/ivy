#!/bin/bash
# Sensure our script doesn't fail silently
set -efu

# Some basic parameters that should rarely rarely change but
# can be specified through environment variables
if [[ -z "${IVY_PREFIX:-}" ]]; then
    IVY_PREFIX="/ivy"
fi
if [[ -z "${DOCKER_IMAGE:-}" ]]; then
    DOCKER_IMAGE="unifyai/ivy:latest"
fi
if [[ -z "${PYTEST_CMD:-}" ]]; then
    PYTEST_CMD=("python3" "-m" "pytest")
else
    echo "WARNING: running eval on: PYTEST_CMD=${PYTEST_CMD}"
    eval PYTEST_CMD="(${PYTEST_CMD})"
fi
if [[ -z "${DOCKER_CMD:-}" ]]; then
    DOCKER_CMD="docker"
fi
if [[ -z "${DOCKER_FLAGS:-}" ]]; then
    DOCKER_FLAGS=("--rm" "-it")
else
    echo "WARNING: running eval on: DOCKER_FLAGS=${DOCKER_FLAGS}"
    eval DOCKER_FLAGS="(${DOCKER_FLAGS})"
fi
if [[ -z "${DEBUG:-}" ]]; then
    DEBUG=0
fi
PWD="$(pwd)"
RAW_ARGS=("$@") # These quotation marks are indispensible to avoid bugs with filenames with spaces

# Check if help is needed
if [[ "-h" == "${1:-}" || "--help" == "${1:-}" ]]; then
    echo "USAGE: $0 [PYTEST OPTIONS AND TESTS]"
    echo "EXAMPLES"
    echo "  $0"
    echo "  $0 -q --no-header --no-summary"
    echo "  $0 ivy_tests/test_ivy/test_functional/test_core/test_general.py"
    echo "  $0 ivy_tests/test_ivy/test_functional/test_core/test_general.py::test_array_equal"
    echo "  $0 -q ivy_tests/test_ivy/test_functional/test_core/test_general.py::test_array_equal --no-header ivy_tests/test_ivy/test_functional/test_core/test_general.py --no-summary"
    echo "Remember to always use relative paths from the repository root"
    exit 1
fi

if (( "${DEBUG}" )); then
    printf -v TMP '%s; ' ${RAW_ARGS[@]}
    echo "RAW_ARGS[@] =", $TMP
fi

# Process the arguments to prepend the paths so they will work inside the docker container
FINAL_ARGS=()
for ARG in ${RAW_ARGS[@]}; do
    if [[ "${ARG::1}" == "-" ]]; then
        # Copy
        FINAL_ARGS+=("${ARG}")
    else
        # Fix paths to test files
        FINAL_ARGS+=("${IVY_PREFIX}/${ARG}")
    fi
done

if (( "${DEBUG}" )); then
    printf -v TMP '%s; ' ${FINAL_ARGS[@]}
    echo "FINAL_ARGS[@] =", $TMP
fi

if (( "${DEBUG}" )); then
    set -x
fi

# These quotation marks are indispensible to avoid bugs with filenames with spaces
"${DOCKER_CMD}" run "${DOCKER_FLAGS[@]}" -v "${PWD}":"${IVY_PREFIX}" "${DOCKER_IMAGE}" "${PYTEST_CMD[@]}" "${FINAL_ARGS[@]}"

if (( "${DEBUG}" )); then
    set +x
fi
