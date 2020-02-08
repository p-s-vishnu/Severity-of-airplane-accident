import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from time import time

# custom scripts
import models.random_forest as rf
import models.classification_models as cm
import models.ensemble as en
from preprocess.scaler import standardize_all, log_transform, power_transform
from sklearn.feature_selection import RFE

train = pd.read_csv("data/train.csv")
train.set_index('Accident_ID', inplace=True)
test = pd.read_csv("data/test.csv")

target = "Severity"
common_cols = ["Violations","Accident_Type_Code","Turbulence_In_gforces","Total_Safety_Complaints","Cabin_Temperature","Max_Elevation","Adverse_Weather_Metric"]

X = train.drop(columns=[target])
X = X.drop(columns=common_cols)
y = train[target]
X_t = test.drop(columns=['Accident_ID'])
X_t = X_t.drop(columns=common_cols)

### Preprocess - Log,Standardize,Exponential, Sqrt
# X.loc[:], X_t.loc[:]  = standardize_all(X, X_t)
# X.loc[:], X_t.loc[:]  = log_transform(X), log_transform(X_t)
# transform_columns = X.columns

n = 0.1
transform_columns = ["Safety_Score","Control_Metric","Days_Since_Inspection"]
X.loc[:, transform_columns] = power_transform(X, n= n, columns=transform_columns)
X_t.loc[:, transform_columns] = power_transform(X_t, n= n, columns=transform_columns)


### Train
start = time()
model = rf.train(X, y)
# cm.classification_report(X,y)
print(f"-----Time: Cross validation----- \n{round(time()-start,2)} seconds")

### Submission - test.csv
# """
output = model.predict(X_t)
result = pd.DataFrame({
    "Accident_ID": test['Accident_ID'],
    "Severity": output
})
out_path = './data/random/Result.csv'
result.to_csv(out_path, index=False)
# """


print("-------- Done ------------")