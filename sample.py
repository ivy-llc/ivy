from ivy.functional.frontends.xgboost import XGBClassifier
import time


def filter_keys():
    dt = {
        "n_estimators": None,
        "objective": "binary:logistic",
        "max_depth": None,
        "max_leaves": None,
        "max_bin": None,
        "grow_policy": None,
        "learning_rate": None,
        "verbosity": None,
        "booster": None,
        "tree_method": None,
        "gamma": None,
        "min_child_weight": None,
        "max_delta_step": None,
        "subsample": None,
        "sampling_method": None,
        "colsample_bytree": None,
        "colsample_bylevel": None,
        "colsample_bynode": None,
        "reg_alpha": None,
        "reg_lambda": None,
        "scale_pos_weight": None,
        "base_score": None,
        "missing": None,
        "num_parallel_tree": None,
        "random_state": None,
        "n_jobs": None,
        "monotone_constraints": None,
        "interaction_constraints": None,
        "importance_type": None,
        "device": None,
        "validate_parameters": None,
        "enable_categorical": False,
        "feature_types": None,
        "max_cat_to_onehot": None,
        "max_cat_threshold": None,
        "multi_strategy": None,
        "eval_metric": None,
        "early_stopping_rounds": None,
        "callbacks": None,
    }
    wrapper_specific = [
        "importance_type",
        "kwargs",
        "missing",
        "n_estimators",
        "use_label_encoder",
        "enable_categorical",
        "early_stopping_rounds",
        "callbacks",
        "feature_types",
    ]
    return {key: dt[key] for key in dt.keys() if key not in wrapper_specific}


xgb = XGBClassifier()
start = time.time()
xgb.get_xgb_params()
end = time.time()
print("Old filtering time: ", end - start)

start = time.time()
filter_keys()
end = time.time()
print("New filtering time: ", end - start)
