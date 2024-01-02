from abc import ABCMeta, abstractmethod
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
)
import copy
from ._criterion import Gini, Criterion
from ._splitter import BestSplitter, Splitter
from ._tree import DepthFirstTreeBuilder, Tree
import ivy
import numbers


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

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        ivy.seed(seed_value=self.random_state)
        n_samples, self.n_features_in_ = X.shape
        y = ivy.atleast_1d(y)
        if y.ndim == 1:
            y = ivy.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]
        y = ivy.copy.copy(y)
        self.classes_ = []
        self.n_classes_ = []
        if self.class_weight is not None:
            ivy.copy.copy(y)
        y_encoded = ivy.zeros(y.shape, dtype=ivy.int32)

        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k] = ivy.unique_inverse(y[:, k])
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_encoded

        self.n_classes_ = ivy.array(self.n_classes_, dtype="int64")

        y = ivy.array(y, dtype="float32")
        max_depth = (
            ivy.iinfo(ivy.int32).max if self.max_depth is None else self.max_depth
        )

        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:
            min_samples_leaf = int(ivy.ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:
            min_samples_split = int(ivy.ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        if self.max_features is None:  # todo: other cases
            max_features = self.n_features_in_
        self.max_features_ = max_features
        assert len(y) == n_samples, "Number of labels does not match number of samples"

        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * ivy.sum(sample_weight)

        self.n_classes_ = ivy.array(self.n_classes_, dtype=ivy.int64)

        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = Gini(self.n_outputs_, self.n_classes_)
        else:
            criterion = copy.deepcopy(criterion)
        splitter = self.splitter
        monotonic_cst = None

        if not isinstance(self.splitter, Splitter):
            splitter = BestSplitter(
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                self.random_state,
                monotonic_cst,
            )
        self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        builder = DepthFirstTreeBuilder(
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            self.min_impurity_decrease,
        )
        builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)
        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
        self._prune_tree()
        return self

    def _prune_tree(self):
        if self.ccp_alpha == 0.0:
            return
        n_classes = ivy.atleast_1d(self.n_classes_)
        pruned_tree = Tree(self.n_features_in_, n_classes, self.n_outputs_)
        self.tree_ = pruned_tree

    def predict(self, X, check_input=True):
        ivy.seed(seed_value=self.random_state)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # Classification

        if self.n_outputs_ == 1:
            return ivy.gather(self.classes_, ivy.argmax(proba, axis=1), axis=0)

        else:
            class_type = self.classes_[0].dtype
            predictions = ivy.zeros((n_samples, self.n_outputs_), dtype=class_type)
            for k in range(self.n_outputs_):
                predictions[:, k] = ivy.gather(
                    self.classes_[k], ivy.argmax(proba[:, k], axis=1), axis=0
                )

            return predictions

    def apply(self, X, check_input=True):
        raise NotImplementedError

    def decision_path(self, X, check_input=True):
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
