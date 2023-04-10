# Copyright 2022 The ml_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import metadata

from absl.testing import absltest
import ml_dtypes


class CustomFloatTest(absltest.TestCase):
  def test_version_matches_package_metadata(self):
    try:
      ml_dtypes_metadata = metadata.metadata("ml_dtypes")
    except ImportError as err:
      raise absltest.SkipTest("Package metadata not found") from err

    metadata_version = ml_dtypes_metadata["version"]
    package_version = ml_dtypes.__version__
    self.assertEqual(metadata_version, package_version)


if __name__ == "__main__":
  absltest.main()
