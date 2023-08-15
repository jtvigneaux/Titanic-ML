import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Local imports
from feature_selection import filter_method, backward_elimination, rfe


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
                                                                  pd.DataFrame, pd.Series,
                                                                  pd.DataFrame, pd.DataFrame):
    # Get features to use
    # # Significance
    clean_train_data, clean_test_data, *_ = rfe(train, pred)
    # Replace NaN
    # for col in clean_test_data.columns:
    #     if col == "PassengerId":
    #         continue
    #     clean_train_data[col] = clean_train_data[col].fillna(
    #         clean_train_data[col].mean())
    #     clean_test_data[col] = clean_test_data[col].fillna(
    #         clean_test_data[col].mean())
    clean_train_data = clean_train_data.fillna(clean_test_data.mean())
    clean_test_data = clean_test_data.fillna(clean_test_data.mean())

    # Result
    y = clean_train_data.Survived
    # Input matrix
    X = clean_train_data.loc[:, clean_train_data.columns != "Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    # Scale the data
    sc = StandardScaler()
    if "Fare" in clean_train_data.columns:
        ct = ColumnTransformer([
            ('somename', sc, ["Fare"])], remainder='passthrough')
        X_train = pd.DataFrame(
            ct.fit_transform(X_train), columns=X_test.columns)
        X_test = pd.DataFrame(ct.transform(X_test), columns=X_test.columns)

    return X_train, y_train, X_test, y_test, X, y, clean_train_data, clean_test_data


def logistic_reg(train: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    X_train, y_train, X_test, y_test, _, _, _,  clean_test_data = pre_process_data(
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


def svm(train: pd.DataFrame, predict: pd.DataFrame, kernel: str = "rbf") -> pd.DataFrame:
    X_train, y_train, X_test, y_test, X, y, _, clean_test_data = pre_process_data(
        train, predict)
    # Classifier
    classifier = SVC(kernel=kernel, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Making the confusion matrix (to evaluate the results)
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))

    # Plotting
    # plot_model(classifier, X, y)

    # Create the result
    test_data = clean_test_data.drop("PassengerId", axis=1)
    predictions = classifier.predict(test_data)
    return pd.DataFrame(
        {"PassengerId": clean_test_data.PassengerId, "Survived": predictions})


def plot_model(clf: SVC, X: pd.DataFrame, y: pd.Series, fignum: int = 1) -> None:
    print(len(X))
    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, zorder=10,
                cmap=plt.cm.Paired, edgecolors="k")

    plt.axis("tight")
    x_min = -5
    x_max = 5
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(
        XX,
        YY,
        Z,
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
        levels=[-0.5, 0, 0.5],
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())


if __name__ == "__main__":
    df_train = load_data("./data/train.csv")
    df_test = load_data("./data/test.csv")

    # Createclean_train_data, clean_test_data, *_ = filter_method(df_train, df_test)
    clean_train_data, clean_test_data, *_ = rfe(df_train, df_test)

    # result = random_forest_model(df_train, df_test)
    # result = logistic_reg(df_train, df_test)
    result = svm(df_train, df_test, kernel="poly")
    # plt.show()
    result.to_csv('submission.csv', index=False)
