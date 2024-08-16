import ivy
from ivy.functional.frontends.xgboost.objective.regression_loss import (
    LogisticRegression,
)
from ivy.functional.frontends.xgboost.linear.updater_coordinate import (
    coordinate_updater,
)
from copy import deepcopy


class GBLinear:
    def __init__(self, params=None, compile=False, cache=None):
        # we start boosting from zero
        self.num_boosted_rounds = 0

        # default parameter
        # xgboost provides other options for it but the way to modify it remains
        # undocumented for Python API
        self.updater = coordinate_updater

        # LogisticRegression corresponds to 'binary:logistic' objective in terms of
        # calculations
        # In xgboost LogisticClassification is used, but it simply subclasses
        # LogisticRegression redefining the method which returns the name of objective
        self.obj = LogisticRegression()

        self.base_score = self.obj.prob_to_margin(params["base_score"])

        # when weights for groups are not provided this equals to a number of instances
        # in data
        # TODO: add weight sum calculation from provided weights, by now always assume
        # default behaviour
        self.num_inst = params["num_instances"]
        self.sum_instance_weight_ = self.num_inst
        self.scale_pos_weight = (
            1.0 if not params["scale_pos_weight"] else params["scale_pos_weight"]
        )

        # set to True, because we're assuming default behaviour for group weights
        self.is_null_weights = True

        self.is_converged_ = False
        self.tolerance = 0.0

        self.num_output_group = params["num_output_group"]
        self.num_feature = params["num_feature"]

        # xgboost stores weights in a vector form, but it was decided to store them as a
        # 2D matrix here it simplifies calculations while math remains the same
        # added 1 in the first dim, because xgboost stores weights and biases jointly
        self.weight = ivy.zeros(
            (self.num_feature + 1, self.num_output_group), dtype=ivy.float32
        )
        # used to calculate convergence(comparing max difference of weights to
        # tolerance)
        self.prev_weight = deepcopy(self.weight)

        # if base margin is None, use base_score instead
        self.base_margin = (
            params["base_margin"] if params["base_margin"] else self.base_score
        )

        # setup lr and denormalize regularization params for updater
        self.learning_rate = params["learning_rate"]
        self.reg_lambda_denorm = self.sum_instance_weight_ * params["reg_lambda"]
        self.reg_alpha_denorm = self.sum_instance_weight_ * params["reg_alpha"]

        # compilation block
        self.compile = compile
        if self.compile:
            # don't enable native compilation for torch, bc it's already fast enough
            # and this only increases the compilation time
            backend_compile = True if ivy.current_backend_str() != "torch" else False
            self._comp_pred = ivy.trace_graph(_pred, backend_compile=backend_compile)
            self._comp_get_gradient = ivy.trace_graph(
                _get_gradient, backend_compile=backend_compile, static_argnums=(0,)
            )
            self._comp_updater = ivy.trace_graph(
                self.updater, backend_compile=backend_compile
            )

            # run each function to compile it
            # this process doesn't affect the training
            pred = self._comp_pred(cache[0], self.weight, self.base_margin)
            gpair = self._comp_get_gradient(
                self.obj, pred, cache[1], self.scale_pos_weight
            )
            self._comp_updater(
                gpair,
                cache[0],
                self.learning_rate,
                self.weight,
                self.num_feature,
                0,
                self.reg_alpha_denorm,
                self.reg_lambda_denorm,
            )

    def boosted_rounds(self):
        return self.num_boosted_rounds

    def model_fitted(self):
        return self.num_boosted_rounds != 0

    def check_convergence(self):
        if self.tolerance == 0.0:
            return False

        elif self.is_converged_:
            return True

        largest_dw = ivy.max(ivy.abs(self.weight - self.prev_weight))
        self.prev_weight = self.weight.copy()

        self.is_converged_ = largest_dw <= self.tolerance
        return self.is_converged_

    # used to obtain raw predictions
    def pred(self, data):
        args = (data, self.weight, self.base_margin)
        if self.compile:
            return self._comp_pred(*args)
        else:
            return _pred(*args)

    def get_gradient(self, pred, label):
        args = (self.obj, pred, label, self.scale_pos_weight)
        if self.compile:
            return self._comp_get_gradient(*args)
        else:
            return _get_gradient(*args)

    def do_boost(self, data, gpair, iter):
        if not self.check_convergence():
            self.num_boosted_rounds += 1
            args = (
                gpair,
                data,
                self.learning_rate,
                self.weight,
                self.num_feature,
                iter,
                self.reg_alpha_denorm,
                self.reg_lambda_denorm,
            )
            if self.compile:
                self.weight = self._comp_updater(*args)
            else:
                self.weight = self.updater(*args)


# --- Helpers --- #
# --------------- #


def _get_gradient(obj, pred, label, scale_pos_weight):
    p = obj.pred_transform(pred)

    # because we assume default behaviour for group weights this always equals to 1
    # ToDo: add handling for group weights case
    w = 1.0

    # group weights for positive class are scaled
    w_scaled = ivy.where(label == 1.0, w * scale_pos_weight, w)

    return ivy.hstack(
        [
            obj.first_order_gradient(p, label) * w_scaled,
            obj.second_order_gradient(p, label) * w_scaled,
        ]
    )


def _pred(dt, w, base):
    return ivy.matmul(dt, w[:-1]) + w[-1] + base
