import ivy
from ivy.functional.frontends.xgboost.objective.regression_loss import (
    LogisticRegression,
)
from ivy.functional.frontends.xgboost.linear.updater_coordinate import (
    coordinate_updater,
)


class GBLinear:
    def __init__(self, params=None):
        # we start boosting from zero
        self.num_boosted_rounds = 0

        # default parameter
        # xgboost provides other options for it but the way to modify it remains undocumented for Python API
        self.updater = coordinate_updater

        # LogisticRegression corresponds to 'binary:logistic' objective in terms of calculations
        # In xgboost LogisticClassification is used, but it simply subclasses LogisticRegression
        # redefining the method which returns the name of objective
        self.obj = LogisticRegression()

        self.base_score = self.obj.prob_to_margin(params["base_score"])

        # when weights for groups are not provided this equals to a number of instances in data
        # ToDo: add weight sum calculation from provided weights, by now always assume default behaviour
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

        # xgboost stores weights in a vector form, but it was decided to store them as a 2D matrix here
        # it simplifies calculations while math remains the same
        # added 1 in the first dim, because xgboost stores weights and biases jointly
        self.weight = ivy.zeros(
            (self.num_feature + 1, self.num_output_group), dtype=ivy.float32
        )
        # used to calculate convergence(comparing max difference of weights to tolerance)
        self.prev_weight = self.weight.copy()

        # if base margin is None, use base_score instead
        self.base_margin = (
            params["base_margin"] if params["base_margin"] else self.base_score
        )

        # setup lr and denormalize regularization params for updater
        self.learning_rate = params["learning_rate"]
        self.reg_lambda_denorm = self.sum_instance_weight_ * params["reg_lambda"]
        self.reg_alpha_denorm = self.sum_instance_weight_ * params["reg_alpha"]

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
        return _pred(data, self.weight, self.base_margin)

    def get_gradient(self, pred, label):
        return _get_gradient(self.obj, pred, label, self.scale_pos_weight)

    def do_boost(self, data, gpair, iter):
        if not self.check_convergence():
            self.num_boosted_rounds += 1
            self.weight = self.updater(
                gpair,
                data,
                self.learning_rate,
                self.weight,
                self.num_feature,
                iter,
                self.reg_alpha_denorm,
                self.reg_lambda_denorm,
            )


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
