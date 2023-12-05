from abc import ABCMeta, abstractmethod
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
)


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
        raise NotImplementedError

    def predict(self, X, check_input=True):
        raise NotImplementedError

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
