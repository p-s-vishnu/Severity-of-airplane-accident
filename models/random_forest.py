from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,RepeatedKFold
import numpy as np
from xgboost import XGBClassifier

# custom module
from models.validate import cross_validate 

params = {
    "random_state"  :   5,
    "n_jobs"        :   -1,
    # "criterion"     : "entropy",
    'n_estimators'  :   930,
    # 'max_features'  :   5,
    'min_samples_split': 3,
    'max_depth'     :   30,
    "oob_score"     :   True,
    "class_weight"  :"balanced"
}

base_params = {
    "random_state":5,
    "n_jobs":-1,
    "class_weight":"balanced",
}

def train(X,y):

    model = RandomForestClassifier(**params)
    # model = XGBClassifier(learning_rate=1,n_jobs=-1,objective="multi:softmax",random_state=5,max_depth=3,n_estimators=1200)
    
    # skf = StratifiedKFold(n_splits=2, shuffle=True,random_state=42)
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.8, random_state=42)
    # rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
    
    # print model metrics
    metrics = cross_validate(sss, X, y, model)

    print("Train & Cross validation".center(40,'-'))
    print(np.mean(metrics,axis=0)*100)

    model.fit(X,y)
    return model

