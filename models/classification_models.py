from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

# models
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC,SVC,NuSVC
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier

# custom packages
from models.validate import cross_validate

def classification_report(X,y):
    """
        Trains and tries out most of the classification models and 
        displays the metric values for the same

        inp: X,y
        out: None
        condition: print metics values
    """
    seed = 5
    models = [
        XGBClassifier(learning_rate=1,n_jobs=-1,objective="multi:softmax"),
        LogisticRegression(random_state=seed,n_jobs=-1,multi_class="multinomial",max_iter=5000,solver="saga",penalty='l1'),
        RidgeClassifier(random_state=seed,max_iter=2000),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        DecisionTreeClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed,n_jobs=-1),
        ExtraTreeClassifier(random_state=seed),
        ExtraTreesClassifier(random_state=seed,n_jobs=-1),
        GradientBoostingClassifier(random_state=seed),
        HistGradientBoostingClassifier(random_state=5),
        AdaBoostClassifier(random_state=seed),
        KNeighborsClassifier(n_neighbors=3),
        NearestCentroid(),
        # RadiusNeighborsClassifier(n_neighbors=3),
        MLPClassifier(random_state=seed),
        # LinearSVC(random_state=seed,max_iter=100),
        SVC(random_state=seed),
        NuSVC(random_state=seed),
        BernoulliNB(),
        GaussianNB(),
        GaussianProcessClassifier(random_state=seed,n_jobs=-1),
        # LabelPropagation(n_jobs=-1),
        # LabelSpreading(n_jobs=-1),
    ]

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.8, random_state=42)

    report = []
    for i,model in enumerate(models):
        # print model metrics
        print(str(model.__class__).center(30,'-'),'\n')
        metrics = cross_validate(sss, X, y, model)
        report.append([str(models[i]), np.mean(metrics,axis=0)[1]*100])

    report = pd.DataFrame(report, columns=['model','score'])
    print(report.sort_values(by="score",ascending=False))
    report.sort_values(by="score",ascending=False).to_csv('classification_report.csv',index=False)
