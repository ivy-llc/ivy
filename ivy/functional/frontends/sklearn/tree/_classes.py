from abc import ABCMeta, abstractmethod
from scipy.sparse import issparse
import ivy
from ..utils.multiclass import check_classification_targets
from ..utils.class_weight import compute_sample_weight
from ..base import is_classifier
from _tree import DTYPE, DOUBLE, Tree, BestFirstTreeBuilder, DepthFirstTreeBuilder
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
)
import numbers
import warnings
from ..utils.validation import _check_sample_weight
from _criterion import Criterion, Gini, Entropy, MSE, MAE, FriedmanMSE, Poisson
from splitter import (
    Splitter,
    BestSplitter,
    RandomSplitter,
    RandomSparseSplitter,
    BestSparseSplitter,
)

CRITERIA_CLF = {
    "gini": Gini,
    "log_loss": Entropy,
    "entropy": Entropy,
}
CRITERIA_REG = {
    "squared_error": MSE,
    "friedman_mse": FriedmanMSE,
    "absolute_error": MAE,
    "poisson": Poisson,
}
DENSE_SPLITTERS = {"best": BestSplitter, "random": RandomSplitter}
SPARSE_SPLITTERS = {
    "best": BestSparseSplitter,
    "random": RandomSparseSplitter,
}


class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def get_depth(self):
        raise NotImplementedError

    def get_n_leaves(self):
        raise NotImplementedError

    def _support_missing_values(self, X):
        raise NotImplementedError

    def _compute_missing_values_in_feature_mask(self, X):
        raise NotImplementedError

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        if check_input:
            check_X_params = dict(
                dtype=DTYPE, accept_sparse="csc", force_all_finite=False
            )
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )

            missing_values_in_feature_mask = (
                self._compute_missing_values_in_feature_mask(X)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != ivy.int64 or X.indptr.dtype != ivy.int64:
                    raise ValueError(
                        "No support for ivy.int64 index based sparse matrices"
                    )

            if self.criterion == "poisson":
                if ivy.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if ivy.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape
        is_classification = is_classifier(self)

        y = ivy.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, ivy.newaxis] that does not.
            y = ivy.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = ivy.copy.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = ivy.copy.copy(y)

            y_encoded = ivy.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = ivy.unique_all(
                    y[:, k], return_inverse=True
                )
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = ivy.array(self.n_classes_, dtype=ivy.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = ivy.ascontiguousarray(y, dtype=DOUBLE)

        max_depth = (
            ivy.iinfo(ivy.int32).max if self.max_depth is None else self.max_depth
        )

        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ivy.ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ivy.ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(ivy.sqrt(self.n_features_in_)))
                    warnings.warn(
                        "`max_features='auto'` has been deprecated in 1.1 "
                        "and will be removed in 1.3. To keep the past behaviour, "
                        "explicitly set `max_features='sqrt'`.",
                        FutureWarning,
                    )
                else:
                    max_features = self.n_features_in_
                    warnings.warn(
                        "`max_features='auto'` has been deprecated in 1.1 "
                        "and will be removed in 1.3. To keep the past behaviour, "
                        "explicitly set `max_features=1.0'`.",
                        FutureWarning,
                    )
            elif self.max_features == "sqrt":
                max_features = max(1, int(ivy.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(ivy.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * ivy.sum(sample_weight)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = ivy.copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                self.random_state,  # TODO: It was just random_state, no idea why it was not recognized!
            )

        if is_classifier(self):
            self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                ivy.array([1] * self.n_outputs_, dtype=ivy.intp),
                self.n_outputs_,
            )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self

    def _validate_X_predict(self, X, check_input):
        raise NotImplementedError

    def predict(self, X, check_input=True):
        raise NotImplementedError

    def apply(self, X, check_input=True):
        raise NotImplementedError

    def decision_path(self, X, check_input=True):
        raise NotImplementedError

    def _prune_tree(self):
        raise NotImplementedError

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        raise NotImplementedError

    @property
    def feature_importances_(self):
        raise NotImplementedError


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def fit(self, X, y, sample_weight=None, check_input=True):
        super()._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        return self

    def predict_proba(self, X, check_input=True):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def _more_tags(self):
        allow_nan = self.splitter == "best" and self.criterion in {
            "gini",
            "log_loss",
            "entropy",
        }
        return {"multilabel": True, "allow_nan": allow_nan}
