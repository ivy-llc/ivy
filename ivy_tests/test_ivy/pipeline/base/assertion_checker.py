import numpy as np
from .runners import TestCaseSubRunnerResult

DEFAULT_ATOL = 1e-06
DEFAULT_RTOL = None
TOLERANCE_DICT = {
    "float16": 1e-2,
    "bfloat16": 1e-2,
    "float32": 1e-5,
    "float64": 1e-5,
    None: 1e-5,
}


class AssertionChecker:
    def __init__(
        self,
        target_results: TestCaseSubRunnerResult,
        ground_truth_results: TestCaseSubRunnerResult,
        backend: str,
        ground_truth_backend: str,
        specific_tolerance_dict: dict,
        rtol: dict,
        atol: dict,
    ):
        self.target_results = target_results
        self.ground_truth_results = ground_truth_results
        self.backend_to_test = backend
        self.ground_truth_backend = ground_truth_backend
        self.specific_tolerance_dict = specific_tolerance_dict
        self.rtol = rtol
        self.atol = atol

    def _get_framework_rtol(self):
        if self.backend_to_test in self.rtol.keys():
            return self.rtol[self.backend_to_test]
        return DEFAULT_RTOL

    def _get_framework_atol(self):
        if self.backend_to_test in self.atol.keys():
            return self.atol[self.backend_to_test]
        return DEFAULT_ATOL

    def _assert_type(self, target_type, ground_truth_type):
        for type, gt_type in zip(target_type, ground_truth_type):
            assert type == gt_type

    def _assert_dtype(self, target_dtype, ground_truth_dtype):
        for dtype, gt_dtype in zip(target_dtype, ground_truth_dtype):
            assert dtype == gt_dtype

    def _assert_device(self, target_device, ground_truth_device):
        for target_dev, gt_dev in zip(target_device, ground_truth_device):
            assert target_dev == gt_dev, (
                f"ground truth backend ({self.ground_truth_backend}) returned array on"
                f" device {gt_dev} but target backend ({self.backend_to_test})"
                f" returned array on device {target_dev}"
            )

    def _assert_same_length(self, target_alements, ground_truth_elements):
        assert len(target_alements) == len(ground_truth_elements), (
            f"The length of results from backend {self.backend_to_test} and ground"
            " truth framework"
            f" {self.ground_truth_backend} does not match\n\nlen(ret_np_flat) !="
            f" len(ret_np_from_gt_flat):\n\nret_np_flat:\n\n{target_alements}\n\n"
            f"ret_np_from_gt_flat:\n\n{ground_truth_elements}"
        )

    def _assert_all_close(self, ret_np, ret_from_gt_np, rtol, atol):
        ret_dtype = str(ret_np.dtype)
        ret_from_gt_dtype = str(ret_from_gt_np.dtype).replace("longlong", "int64")
        assert ret_dtype == ret_from_gt_dtype, (
            f"the ground truth framework {self.ground_truth_backend} returned a"
            f" {ret_from_gt_dtype} datatype while the backend"
            f" {self.backend_to_test} returned a {ret_dtype} datatype"
        )
        # TODO eanble
        # if ivy.is_ivy_container(ret_np) and ivy.is_ivy_container(ret_from_gt_np):
        #     ivy.Container.cont_multi_map(assert_all_close, [ret_np, ret_from_gt_np])
        # else:
        if ret_np.dtype == "bfloat16" or ret_from_gt_np.dtype == "bfloat16":
            ret_np = ret_np.astype("float64")
            ret_from_gt_np = ret_from_gt_np.astype("float64")
        assert np.allclose(
            np.nan_to_num(ret_np), np.nan_to_num(ret_from_gt_np), rtol=rtol, atol=atol
        ), (
            f" the results from backend {self.backend_to_test} "
            f"and ground truth framework {self.ground_truth_backend} "
            f"do not match\n {ret_np}!={ret_from_gt_np} \n\n"
        )

    def _assert_equal_elements(self, target_elements, ground_truth_elements):
        rtol = self._get_framework_rtol() if isinstance(self.rtol, dict) else self.rtol
        atol = self._get_framework_atol() if isinstance(self.atol, dict) else self.atol
        if self.specific_tolerance_dict is not None:
            for ret_np, ret_np_from_gt in zip(target_elements, ground_truth_elements):
                dtype = str(ret_np_from_gt.dtype)
                if self.specific_tolerance_dict.get(dtype) is not None:
                    rtol = self.specific_tolerance_dict.get(dtype)
                else:
                    rtol = TOLERANCE_DICT.get(dtype, 1e-03) if rtol is None else rtol
                self._assert_all_close(
                    ret_np,
                    ret_np_from_gt,
                    rtol=rtol,
                    atol=atol,
                )
        elif rtol is not None:
            for ret_np, ret_np_from_gt in zip(target_elements, ground_truth_elements):
                self._assert_all_close(
                    ret_np,
                    ret_np_from_gt,
                    rtol=rtol,
                    atol=atol,
                )
        else:
            for ret_np, ret_np_from_gt in zip(target_elements, ground_truth_elements):
                rtol = TOLERANCE_DICT.get(str(ret_np_from_gt.dtype), 1e-03)
                self._assert_all_close(
                    ret_np,
                    ret_np_from_gt,
                    rtol=rtol,
                    atol=atol,
                )

    def check_assertions(self):
        self._assert_dtype(self.target_results.dtype, self.ground_truth_results.dtype)
        self._assert_type(self.target_results.type, self.ground_truth_results.type)
        self._assert_device(
            self.target_results.device, self.ground_truth_results.device
        )
        self._assert_same_length(
            self.target_results.flatten_elements_np,
            self.ground_truth_results.flatten_elements_np,
        )
        self._assert_equal_elements(
            self.target_results.flatten_elements_np,
            self.ground_truth_results.flatten_elements_np,
        )
