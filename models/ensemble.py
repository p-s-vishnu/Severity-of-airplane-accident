from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,StackingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

import numpy as np

from models.validate import cross_validate

params = {
    "random_state"  :   5,
    "n_jobs"        :   -1,
    'n_estimators'  :   4000,
    # 'max_features'  :   5,
    'min_samples_split': 3,
    'max_depth'     :   40,
    "oob_score"     :   True,
    # "class_weight"  :"balanced"
}

grad_params = {
    "random_state"  :   5,
    "learning_rate" :   1,
    # "n_estimators"  : 10,
    # "max_depth"     : 1,
}

hist_params = {
    "random_state"  : 5,
    "max_iter"      : 300,
    "learning_rate" : 0.05,
    "max_leaf_nodes": 20,
    # "max_depth"     : 40,
    # "min_samples_leaf"  :   1,
    # "l2_regularization" :   1,
    # "n_iter_no_change":  100,
    # "warm_start"    : True,
    # "scoring"       : "f1_weighted"
}
xgb_params = {
    "learning_rate" :   1,
    "n_jobs"        :   -1,
    "objective"     :   "multi:softmax",
    "random_state"  : 5,
    "max_depth"     : 40,
    "n_estimators"  :4000
    }


def train(X,y):

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.8, random_state=42)

    # model = HistGradientBoostingClassifier(**hist_params)
    # model = GradientBoostingClassifier(**grad_params)
    # model = XGBClassifier(**xgb_params)

    # """    
    estimators = [
        ("RandomForest"         ,RandomForestClassifier(**params)),
        # ("HistGradientBoosting" ,HistGradientBoostingClassifier(**hist_params)),
        ("Quadrant"             ,QuadraticDiscriminantAnalysis()),
        ("XGB", XGBClassifier(**xgb_params))
    ]
    model = StackingClassifier(
        estimators=estimators,
        n_jobs=-1
    )
    # """
    print("Train & Cross validation".center(40,'-'))
    print(np.mean(cross_validate(sss,X,y,model), axis=0)*100)
    model.fit(X,y)
    # print(model.n_iter_)
    return model
