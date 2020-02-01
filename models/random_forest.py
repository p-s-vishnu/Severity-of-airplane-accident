from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.metrics import f1_score
import tqdm
import numpy as np

params = {
    "random_state"  :5,
    "n_jobs"        :-1,
    'n_estimators'  : 5000,
    'max_features': None,
    'min_samples_split': 3,
    'max_depth': 50,
    "oob_score":True
}

base_params = {
    "random_state":5,
    "n_jobs":-1
}

def train(X,y):

    model = RandomForestClassifier(**params)
    skf = StratifiedKFold(shuffle=True,random_state=42)
    
    # print model metrics
    metrics = []
    for train_index, test_index in tqdm.tqdm(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_score = f1_score(y_train,y_train_pred,average="weighted")
        cross_score = f1_score(y_test,y_test_pred,average="weighted")

        metrics.append([train_score,cross_score])

    print("Train & Cross validation".center(40,'-'))
    print(np.mean(metrics,axis=0)*100)

    model.fit(X,y)
    return model