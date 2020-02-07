from sklearn.metrics import f1_score
import tqdm


def cross_validate(splitter, X, y, model):
    metrics = []
    for train_index, test_index in tqdm.tqdm(splitter.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_score = f1_score(y_train, y_train_pred, average="weighted")
        cross_score = f1_score(y_test, y_test_pred, average="weighted")

        metrics.append([train_score, cross_score])
        return metrics
