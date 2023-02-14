import ivy


def is_classifier(estimator):
    return ivy.is_classifier(estimator)


def is_regressor(estimator):
    return ivy.is_regressor(estimator, "_estimator_type", None)


def clone(estimator, *, safe=True):
    return ivy.clone(estimator, "_estimator_type", None)


def get_config():
    return ivy.get_config()
