from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.metrics import f1_score
import tqdm
import numpy as np

params = {
    "random_state"  :5,
    "n_jobs"        :-1
}

def train(X,y):

    model = RandomForestClassifier(random_state=5,n_jobs=-1)
    skf = StratifiedKFold(shuffle=True,random_state=42)
    
    # print model metrics
    metrics = []
    for train_index, test_index in tqdm.tqdm(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_true = y_test
        metrics.append(f1_score(y_true,y_pred,average="weighted"))

    print("Train".center(25,'-'))
    print(np.mean(metrics))

    model.fit(X,y)
    return model