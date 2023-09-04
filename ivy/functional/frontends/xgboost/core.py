import ivy
from ivy.func_wrapper import with_unsupported_dtypes


class DMatrix:
    def __init__(
        self,
        data,
        label=None,
        *,
        weight=None,
        base_margin=None,
        missing=None,
        silent=False,
        feature_names=None,
        feature_types=None,
        nthread=None,
        group=None,
        qid=None,
        label_lower_bound=None,
        label_upper_bound=None,
        feature_weights=None,
        enable_categorical=False,
    ):
        self.data = ivy.array(data) if not isinstance(data, ivy.Array) else data
        self.label = (
            ivy.array(label) if (not isinstance(label, ivy.Array) and label) else label
        )

        self.weight = (
            (
                ivy.array(weight)
                if (not isinstance(weight, ivy.Array) and weight)
                else weight
            ),
        )
        self.base_margin = (
            (
                ivy.array(base_margin)
                if (not isinstance(base_margin, ivy.Array) and base_margin)
                else base_margin
            ),
        )
        self.missing = (missing,)
        self.silent = (silent,)
        self.feature_names = (feature_names,)
        self.feature_types = (feature_types,)
        self.nthread = (nthread,)
        self.group = (
            ivy.array(group) if (not isinstance(group, ivy.Array) and group) else group,
        )
        self.qid = (
            ivy.array(qid) if (not isinstance(qid, ivy.Array) and qid) else qid,
        )
        self.label_lower_bound = (
            (
                ivy.array(label_lower_bound)
                if (not isinstance(label_lower_bound, ivy.Array) and label_lower_bound)
                else label_lower_bound
            ),
        )
        self.label_upper_bound = (
            (
                ivy.array(label_upper_bound)
                if (not isinstance(label_upper_bound, ivy.Array) and label_upper_bound)
                else label_upper_bound
            ),
        )
        self.feature_weights = (
            (
                ivy.array(feature_weights)
                if (not isinstance(feature_weights, ivy.Array) and feature_weights)
                else feature_weights
            ),
        )
        self.enable_categorical = enable_categorical

    @with_unsupported_dtypes(
        {"1.7.6 and below": ("bfloat16", "complex64", "complex128")}, "xgboost"
    )
    def num_row(self):
        return ivy.shape(self.data)[0]

    @with_unsupported_dtypes(
        {"1.7.6 and below": ("bfloat16", "complex64", "complex128")}, "xgboost"
    )
    def num_col(self):
        return ivy.shape(self.data)[1]
