import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Local imports
from feature_selection import filter_method


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def gender_solution() -> pd.DataFrame:
    """Based on the gender of the person, determines the survival

    Returns:
        pd.DataFrame: result of test data
    """
    train = pd.read_csv("./data/test.csv")
    train["Survived"] = np.where(train.Sex == "female", 1, 0)
    return train[["PassengerId", "Survived"]]


def random_forest_model(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Model based on random forest model (constructing trees)

    Args:
        train (pd.DataFrame): train dataset
        test (pd.DataFrame): test dataset

    Returns:
        pd.DataFrame: results
    """
    # Result
    y = train["Survived"]
    # Input matrix
    columns = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train[columns], drop_first=True)
    X_test = pd.get_dummies(test[columns], drop_first=True)

    # Build 100 trees
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    # Predict test data
    predictions = model.predict(X_test)

    return pd.DataFrame(
        {"PassengerId": test.PassengerId, "Survived": predictions})


def pre_process_data(train: pd.DataFrame, pred: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame,
                                                                  pd.DataFrame, pd.DataFrame,
                                                                  pd.DataFrame):
    # Get features to use
    # # Significance
    clean_train_data, clean_test_data = filter_method(train, pred)
    # Result
    y = clean_train_data.Survived
    # Input matrix
    X = clean_train_data.loc[:, clean_train_data.columns != "Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    # Scale the data
    sc = StandardScaler()
    # X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
    # X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

    return X_train, y_train, X_test, y_test, clean_test_data


def logistic_reg(train: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    X_train, y_train, X_test, y_test, clean_test_data = pre_process_data(
        train, predict)
    # Create classifier
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Making the confusion matrix (to evaluate the results)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    # Create the result
    predictions = classifier.predict(clean_test_data)
    return pd.DataFrame(
        {"PassengerId": predict.PassengerId, "Survived": predictions})


def svm(train: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    X_train, y_train, X_test, y_test, clean_test_data = pre_process_data(
        train, predict)
    # clean_train_data, clean_test_data = filter_method(train, predict)
    # # Result
    # y = clean_train_data.Survived
    # # Input matrix
    # X = clean_train_data.loc[:, clean_train_data.columns != "Survived"]
    # Classifier
    classifier = SVC(kernel='rbf', C=2, gamma=5, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Making the confusion matrix (to evaluate the results)
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))

    # Create the result
    predictions = classifier.predict(clean_test_data)
    return pd.DataFrame(
        {"PassengerId": predict.PassengerId, "Survived": predictions})


if __name__ == "__main__":
    df_train = load_data("./data/train.csv")
    df_test = load_data("./data/test.csv")

    clean_train_data, clean_test_data = filter_method(df_train, df_test)
    # Result
    y = clean_train_data.Survived
    # Input matrix
    X = clean_train_data.loc[:, clean_train_data.columns != "Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # Scale the data
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

    # result = random_forest_model(df_train, df_test)
    # result = logistic_reg(df_train, df_test)
    result = svm(df_train, df_test)
    result.to_csv('submission.csv', index=False)
