import ivy
from ivy.functional.frontends.sklearn.base import BaseEstimator as XGBModelBase
from ivy.functional.frontends.sklearn.base import ClassifierMixin as XGBClassifierBase
from .training import train
from .core import Booster


class XGBModel(XGBModelBase):
    def __init__(
        self,
        max_depth=None,
        max_leaves=None,
        max_bin=None,
        grow_policy=None,
        learning_rate=None,
        n_estimators=None,
        verbosity=None,
        objective=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        sampling_method=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        random_state=None,
        missing=None,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type=None,
        device=None,
        validate_parameters=None,
        enable_categorical=False,
        feature_types=None,
        max_cat_to_onehot=None,
        max_cat_threshold=None,
        multi_strategy=None,
        eval_metric=None,
        early_stopping_rounds=None,
        callbacks=None,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.objective = objective

        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.device = device
        self.validate_parameters = validate_parameters
        self.enable_categorical = enable_categorical
        self.feature_types = feature_types
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.multi_strategy = multi_strategy
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        self.compiled = False

        if kwargs:
            self.kwargs = kwargs

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_Booster")

    def get_booster(self):
        """Get the underlying xgboost Booster of this model. This will raise an
        exception when fit was not called.

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if not self.__sklearn_is_fitted__():
            raise TypeError("need to call fit or load_model beforehand")
        return self._Booster

    def get_params(self, deep=True):
        params = self.__dict__

        # if kwargs is a dict, update params accordingly
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)

        # take random_state into account only if it's an integer
        if isinstance(params["random_state"], int):
            ivy.seed(seed_value=params["random_state"])

        return params

    def get_xgb_params(self):
        """Get xgboost specific parameters."""
        params = self.get_params()

        # Parameters that should not go into native learner.
        wrapper_specific = {
            "importance_type",
            "kwargs",
            "missing",
            "n_estimators",
            "use_label_encoder",
            "enable_categorical",
            "early_stopping_rounds",
            "callbacks",
            "feature_types",
        }
        filtered = {}
        for k, v in params.items():
            if k not in wrapper_specific and not callable(v):
                filtered[k] = v

        return filtered

    def get_num_boosting_rounds(self):
        """Gets the number of xgboost boosting rounds."""
        # 100 is the default number of boosting rounds
        return 100 if not self.n_estimators else self.n_estimators

    def compile(self, X, y):
        # set compiled flag
        self.compiled = True

        # instantiate Booster and compile funcs involved in calculations for training
        params = self.get_xgb_params()
        self._Booster = Booster(params, cache=[X, y], compile=True)

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        xgb_model=None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
        feature_weights=None,
        callbacks=None,
    ):
        """Fit gradient boosting model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X
            Feature matrix.

            When the ``tree_method`` is set to ``hist``, internally, the
            `QuantileDMatrix` will be used instead of the `DMatrix`
            for conserving memory. However, this has performance implications when the
            device of input data is not matched with algorithm. For instance, if the
            input is a numpy array on CPU but ``cuda`` is used for training, then the
            data is first processed on CPU then transferred to GPU.
        y
            Labels.
        sample_weight
            instance weights.
        base_margin
            global bias for each instance.
        eval_set
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_metric
            str, list of str, or callable, optional(deprecated in fit method).
        early_stopping_rounds
            int(deprecated in fit method).
        verbose
            If `verbose` is True and an evaluation set is used, the evaluation metric
            measured on the validation set is printed to stdout at each boosting stage.
            If `verbose` is an integer, the evaluation metric is printed at each
            `verbose` boosting stage. The last boosting stage / the boosting stage found
            by using `early_stopping_rounds` is also printed.
        xgb_model
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set
            A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
            object storing instance weights for the i-th validation set.
        base_margin_eval_set
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.
        callbacks
            (deprecated in fit method).
        """
        # skip all the validation as we're interested in calculations for now
        # ToDo: add handling for custom objective
        if self.compiled:
            for i in range(self.get_num_boosting_rounds()):
                self._Booster.update(X, y, i)
        else:
            params = self.get_xgb_params()
            self._Booster = train(params, X, y, self.get_num_boosting_rounds())

        return self

    def predict(
        self,
        X,
        output_margin=False,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
    ):
        """
        Parameters
        ----------
        X
            Data to predict with.
        output_margin
            Whether to output the raw untransformed margin value.
        validate_features
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin
            Margin added to prediction.
        iteration_range
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying ``iteration_range=(10,
            20)``, then only the forests built during [10, 20) (half open set) rounds
            are used in this prediction.

        Returns
        -------
        prediction

        """
        # skip the validation, as for now we simply call the predict method of
        # underlying booster
        return self.get_booster().predict(
            data=X,
            iteration_range=iteration_range,
            output_margin=output_margin,
            validate_features=validate_features,
        )


class XGBClassifier(XGBModel, XGBClassifierBase):
    # as for now simply calls the init method of a parent class, because we implement a
    # minimal subset of functionality
    def __init__(self, *, objective="binary:logistic", **kwargs):
        super().__init__(objective=objective, **kwargs)
