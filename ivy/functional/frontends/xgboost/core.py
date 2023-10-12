import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from .gbm import GBLinear


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


class Booster:
    def __init__(self, params=None, cache=None, model_file=None, compile=False):
        # cache[0] refers to input data while cache[1] refers to input target
        n_feat = cache[0].shape[1]
        n_inst = cache[0].shape[0]
        n_output_group = ivy.unique_values(cache[1]).shape[0]

        # by default xgboost calculates the mean of a target if base_score is not
        # provided
        params["base_score"] = (
            cache[1].mean() if not params["base_score"] else params["base_score"]
        )

        # add num_feature, num_target and num_instances to params
        params.update(
            {
                "num_feature": n_feat,
                "num_output_group": n_output_group - 1,
                "num_instances": n_inst,
            }
        )

        # create gbm(as for now only gblinear booster is available)
        self.gbm = GBLinear(params, compile=compile, cache=cache)
        self.compile = compile
        if self.compile:
            self._comp_binary_prediction = ivy.trace_graph(
                _binary_prediction, backend_compile=True, static_argnums=(0,)
            )

            # invoke function to get its compiled version
            self._comp_binary_prediction(self.gbm.obj, cache[1])

    def update(self, dtrain, dlabel, iteration, fobj=None):
        """Update for one iteration, with objective function calculated
        internally. This function should not be called directly by users.

        Parameters
        ----------
        dtrain
            Training data.
        dlabel
            Training labels.
        iteration
            Number of current iteration.
        fobj
            Custom objective.
        """
        # ToDo: add support for custom objective
        pred = self.gbm.pred(dtrain)
        gpair = self.gbm.get_gradient(pred, dlabel)
        self.gbm.do_boost(dtrain, gpair, iteration)

    def predict(
        self,
        data,
        output_margin=False,
        pred_leaf=False,
        pred_contribs=False,
        approx_contribs=False,
        pred_interactions=False,
        validate_features=True,
        training=False,
        iteration_range=(0, 0),
        strict_shape=False,
    ):
        """Predict with data. The full model will be used unless
        `iteration_range` is specified, meaning user have to either slice the
        model or use the ``best_iteration`` attribute to get prediction from
        best model returned from early stopping.

        Parameters
        ----------
        data
            The array storing the input.
        output_margin
            Whether to output the raw untransformed margin value.
        pred_leaf
            When this option is on, the output will be a matrix of (nsample,
            ntrees) with each record indicating the predicted leaf index of
            each sample in each tree.  Note that the leaf index of a tree is
            unique per tree, so you may find leaf 1 in both tree 1 and tree 0.
        pred_contribs
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1) with each record indicating the feature contributions
            (SHAP values) for that prediction. The sum of all feature
            contributions is equal to the raw untransformed margin value of the
            prediction. Note the final column is the bias term.
        approx_contribs
            Approximate the contributions of each feature.  Used when ``pred_contribs``
            or ``pred_interactions`` is set to True.  Changing the default of this
            parameter (False) is not recommended.
        pred_interactions
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1, nfeats + 1) indicating the SHAP interaction values for
            each pair of features. The sum of each row (or column) of the
            interaction values equals the corresponding SHAP value (from
            pred_contribs), and the sum of the entire matrix equals the raw
            untransformed margin value of the prediction. Note the last row and
            column correspond to the bias term.
        validate_features
            When this is True, validate that the Booster's and data's
            feature_names are identical.  Otherwise, it is assumed that the
            feature_names are the same.
        training
            Whether the prediction value is used for training.  This can effect `dart`
            booster, which performs dropouts during training iterations but use all
            trees for inference. If you want to obtain result with dropouts, set this
            parameter to `True`.  Also, the parameter is set to true when obtaining
            prediction for custom objective function.
        iteration_range
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction. Unsupported for gblinear booster.
        strict_shape
            When set to True, output shape is invariant to whether classification is
            used.
            For both value and margin prediction, the output shape is (n_samples,
            n_groups), n_groups == 1 when multi-class is not used.  Default to False, in
            which case the output shape can be (n_samples, ) if multi-class is not used.

        Returns
        -------
        prediction : ivy array
        """
        # currently supports prediction for binary task
        # get raw predictions
        pred = self.gbm.pred(data)
        args = (self.gbm.obj, pred)

        if self.compile:
            return self._comp_binary_prediction(*args)
        else:
            return _binary_prediction(*args)


# --- Helpers --- #
# --------------- #


def _binary_prediction(obj, raw_pred):
    # apply activation function
    pred = obj.pred_transform(raw_pred)
    # apply probability thresholding
    return ivy.where(pred >= 0.5, 1.0, 0.0)
