from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
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



def train(X,y):

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.8, random_state=42)

    model = BaggingClassifier(RandomForestClassifier(**params),random_state=5,n_estimators=15,
                            max_samples=0.5, max_features=0.5,n_jobs=-1,verbose=1)
    print("Train & Cross validation".center(40,'-'))
    print(np.mean(cross_validate(sss,X,y,model), axis=0)*100)
    model.fit(X,y)

    return model
