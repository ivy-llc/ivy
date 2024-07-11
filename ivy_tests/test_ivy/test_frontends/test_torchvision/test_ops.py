# global
import numpy as np
from hypothesis import strategies as st


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _nms_helper(draw, batched=False):
    img_width = draw(st.integers(250, 1250))
    img_height = draw(st.integers(250, 1250))
    num_boxes = draw(st.integers(5, 50))
    bbox = {}
    for _ in range(num_boxes):
        x1 = draw(st.integers(0, img_width - 20))
        w = draw(st.integers(5, img_width - x1))
        y1 = draw(st.integers(0, img_height - 20))
        h = draw(st.integers(5, img_height - y1))
        bbox[(x1, y1, x1 + w, y1 + h)] = draw(st.floats(0.1, 0.7))
    iou_threshold = draw(st.floats(0.2, 0.5))
    idxs = None
    if batched:
        bbox_len = len(bbox)
        num_of_categories = draw(st.integers(1, max(bbox_len // 2, 2)))
        idxs = np.arange(num_of_categories)
        idxs = np.random.choice(idxs, size=bbox_len)
    return (
        ["float32", "float32"],
        np.array(list(bbox.keys()), dtype=np.float32),
        np.array(list(bbox.values()), dtype=np.float32),
        iou_threshold,
        idxs,
    )


@st.composite
def _roi_align_helper(draw):
    dtype = draw(helpers.get_dtypes("valid"))[0]
    N = draw(st.integers(1, 5))
    C = draw(st.integers(1, 5))
    H = W = draw(st.integers(5, 20))

    img_width = img_height = draw(st.integers(50, 100))

    spatial_scale = H / img_height

    output_size = draw(st.integers(H - 2, H + 5))

    sampling_ratio = draw(st.one_of(st.just(-1), st.integers(1, 3)))

    aligned = draw(st.booleans())
    input = draw(
        helpers.array_values(
            dtype=dtype,
            shape=(N, C, H, W),
            min_value=-3,
            max_value=3,
        )
    )
    bbox = {}
    for i in range(N):
        num_boxes = draw(st.integers(1, 5))
        for _ in range(num_boxes):
            x1 = draw(st.integers(0, img_width - 20))
            w = draw(st.integers(5, img_width - x1))
            y1 = draw(st.integers(0, img_height - 20))
            h = draw(st.integers(5, img_height - y1))
            bbox[(i, x1, y1, x1 + w, y1 + h)] = 1

    return (
        [dtype],
        input,
        np.array(list(bbox.keys()), dtype=dtype).reshape((-1, 5)),
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
    )


# --- Main --- #
# ------------ #


# batched_nms
@handle_frontend_test(
    fn_tree="torchvision.ops.batched_nms",
    dts_boxes_scores_iou_idxs=_nms_helper(batched=True),
    test_with_out=st.just(False),
)
def test_torchvision_batched_nms(
    *,
    dts_boxes_scores_iou_idxs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dts, boxes, scores, iou, idxs = dts_boxes_scores_iou_idxs
    helpers.test_frontend_function(
        input_dtypes=dts,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        boxes=boxes,
        scores=scores,
        idxs=idxs,
        iou_threshold=iou,
    )


# box_area
@handle_frontend_test(
    fn_tree="torchvision.ops.box_area",
    boxes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(helpers.ints(min_value=1, max_value=5), st.just(4)),
    ),
)
def test_torchvision_box_area(
    *,
    boxes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, boxes = boxes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        boxes=boxes[0],
    )


@handle_frontend_test(
    fn_tree="torchvision.ops.box_iou",
    boxes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(helpers.ints(min_value=1, max_value=5), st.just(4)),
        num_arrays=2,
    ),
)
def test_torchvision_box_iou(
    *,
    boxes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, boxes = boxes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        boxes1=boxes[0],
        boxes2=boxes[1],
    )


@handle_frontend_test(
    fn_tree="torchvision.ops.clip_boxes_to_image",
    boxes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.tuples(helpers.ints(min_value=1, max_value=5), st.just(4)),
    ),
    size=st.tuples(
        helpers.ints(min_value=1, max_value=256),
        helpers.ints(min_value=1, max_value=256),
    ),
)
def test_torchvision_clip_boxes_to_image(
    *,
    boxes,
    size,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, boxes = boxes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        boxes=boxes[0],
        size=size,
    )


# nms
@handle_frontend_test(
    fn_tree="torchvision.ops.nms",
    dts_boxes_scores_iou=_nms_helper(),
    test_with_out=st.just(False),
)
def test_torchvision_nms(
    *,
    dts_boxes_scores_iou,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dts, boxes, scores, iou, _ = dts_boxes_scores_iou
    helpers.test_frontend_function(
        input_dtypes=dts,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        boxes=boxes,
        scores=scores,
        iou_threshold=iou,
    )


# remove_small_boxes
@handle_frontend_test(
    fn_tree="torchvision.ops.remove_small_boxes",
    boxes=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.tuples(helpers.ints(min_value=1, max_value=5), st.just(4)),
    ),
    min_size=helpers.floats(
        min_value=0.0,
        max_value=10,
        small_abs_safety_factor=2,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torchvision_remove_small_boxes(
    *,
    boxes,
    min_size,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, boxes = boxes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        boxes=boxes[0],
        min_size=min_size,
    )


# roi_align
@handle_frontend_test(
    fn_tree="torchvision.ops.roi_align",
    inputs=_roi_align_helper(),
    test_with_out=st.just(False),
)
def test_torchvision_roi_align(
    *,
    inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, input, boxes, output_size, spatial_scale, sampling_ratio, aligned = inputs
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        boxes=boxes,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
        rtol=1e-5,
        atol=1e-5,
    )
