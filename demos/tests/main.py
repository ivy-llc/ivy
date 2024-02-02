from jupyter_client import KernelManager
import unittest
import os
import argparse

# local
from helpers import *


class NotebookTest(unittest.TestCase):
    path = None

    def __init__(self, *args, **kwargs):
        self.file, self.config = fetch_notebook_and_configs(NotebookTest.path)
        self.invert_config()
        super().__init__(*args, **kwargs)

    @classmethod
    def setUp(cls):
        cls.km = KernelManager()
        cls.km.start_kernel(
            extra_arguments=["--pylab=inline"], stderr=open(os.devnull, "w")
        )
        cls.kc = cls.km.blocking_client()
        cls.kc.start_channels()
        cls.kc.execute_interactive("import os;os.environ['IVY_ROOT']='.ivy'")

    @classmethod
    def tearDown(cls):
        cls.kc.stop_channels()
        cls.km.shutdown_kernel()
        del cls.km

    def invert_config(self):
        inverted_config = dict()
        for test_type in self.config:
            for cell_number in self.config[test_type]:
                inverted_config[cell_number] = test_type
        self.config = inverted_config

    def test_notebook(self):
        cells_to_ignore = []
        for cell_number, cell_content in enumerate(self.file.cells):
            outputs = []
            print("\n==========================================")
            print(f"cell number : {cell_number + 1}")
            print(f"cell source : {cell_content.source}")
            if (
                cell_content.cell_type != "code"
                or cell_number + 1 in cells_to_ignore
                or self.config.get(cell_number + 1) == "install"
            ):
                continue
            try:
                self.kc.execute_interactive(
                    cell_content.source,
                    output_hook=lambda msg: record_output(
                        msg, outputs, cell_content.execution_count
                    ),
                )
                for output in outputs:
                    if output["output_type"] in ("pyerror", "error"):
                        raise RuntimeError(
                            "runtime output throws an error -: "
                            f"{output['ename']} with value -: {output['evalue']}"
                        )
            except Exception as e:
                self.fail(f"Failed to run cell {cell_number + 1}: {repr(e)}")

            if self.config and cell_number + 1 in self.config:
                print(f"config : {self.config[cell_number + 1]}")

                test_output, test_execution_count = consolidate(outputs)
                cell_output, _ = consolidate(cell_content.outputs)
                cell_execution_count = cell_content.execution_count
                next_test_output, next_cell_output = "", ""

                if (
                    self.config[cell_number + 1].startswith("benchmark")
                    and cell_number + 2 in self.config
                    and self.config[cell_number + 2].startswith("benchmark")
                ):
                    next_cell_content = self.file.cells[cell_number + 1]
                    next_outputs = []
                    try:
                        self.kc.execute_interactive(
                            next_cell_content.source,
                            output_hook=lambda msg: record_output(
                                msg, next_outputs, next_cell_content.execution_count
                            ),
                            timeout=1000,
                        )
                        for next_output in next_outputs:
                            if next_output["output_type"] in ("pyerror", "error"):
                                raise RuntimeError(
                                    "runtime output throws an error -: "
                                    f"{next_output['ename']}\n with value -: {next_output['evalue']} and it "
                                )
                    except Exception as e:
                        self.fail(f"Failed to run cell {cell_number + 2}: {repr(e)}")
                    next_test_output, _ = consolidate(next_outputs)
                    cells_to_ignore.append(cell_number + 2)
                    next_cell_output, _ = consolidate(next_cell_content.outputs)

                print(f"test output : \n{test_output}")
                print(f"cell output : \n{cell_output}")
                print(f"next test output : \n{next_test_output}")
                print(f"next cell output : \n{next_cell_output}")

                value_test(
                    test_obj=self,
                    test_output=test_output,
                    test_execution_count=test_execution_count,
                    cell_output=cell_output,
                    cell_execution_count=cell_execution_count,
                    config=self.config[cell_number + 1],
                    next_test_output=next_test_output,
                    next_cell_output=next_cell_output,
                )


class IterativeTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return IterativeTestResult(self.stream, self.descriptions, self.verbosity)


class IterativeTestResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.stream.writeln(f"Running test: {test.id()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the notebook file")
    args = parser.parse_args()

    root_path = os.sep.join(__file__.split(os.sep)[:-2])
    NotebookTest.path = os.path.join(root_path, args.path)
    print(f"path : {NotebookTest.path}")

    suite = unittest.TestLoader().loadTestsFromTestCase(NotebookTest)
    runner = IterativeTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        exit(0)  # Tests passed
    else:
        exit(1)  # Tests failed
