#!/bin/bash

mkdir -p .hypothesis
# shellcheck disable=SC2046
core_tests=( $(ls -d /ivy/ivy_tests/test_ivy/test_functional/test_core/test*) )
echo $PWD
echo $core_tests
# docker run --rm -v IVY_BACKEND="$1"  -v $(pwd):/ivy -v $(pwd)/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest ivy/ivy_tests/test_ivy/test_functional/test_core/$2

for i in $core_tests
do
  docker run --rm -v IVY_BACKEND="$1"  -v $(pwd):/ivy -v $(pwd)/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest $i &
done

wait
echo "All done"
