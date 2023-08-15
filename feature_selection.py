""" Methods to choose which are the relevant features to use in the model.
"""
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def filter_method(train: pd.DataFrame, test: pd.DataFrame = None) -> (pd.DataFrame, pd.DataFrame, list[str]):
    """calculate the columns to be considered using the filter method
    The filtering here is done using correlation matrix and it is done 
    using Pearson correlation.

    Args:
        train (pd.DataFrame): train dataset

    Returns:
        pd.DataFrame: important columns
    """
    columns = ["Pclass", "Sex", "SibSp", "Fare", "Parch", "Age"]
    features = pd.get_dummies(train[[*columns, "Survived"]], drop_first=True)
    test_df = pd.get_dummies(test[columns], drop_first=True)

    # Calculate the correlations
    cor = features.corr()
    # OPTIONAL - plot correlation
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()

    # Min correlation to be considerd
    min_corr = 0.1
    # Correlations of the target column (survived)
    cor_target = abs(cor.Survived)
    # Select relevant features (more than abs(x) correlation)
    relevant_features = cor_target[cor_target > min_corr]
    final_columns = relevant_features.index.drop("Survived").to_list()
    # final_columns = ["Pclass", "Sex_male", "Age", "Fare"]

    # Calculate the significance between relevant features
    # Less than min_corr (both are considered)
    # print(features[["Pclass", "Sex_male"]].corr())

    # Show data
    # print("Show data")
    # plt.figure(figsize=(5, 4))
    # seaborn_plot = plt.axes(projection='3d')
    # print(type(seaborn_plot))
    # seaborn_plot.scatter3D(
    #     features.Pclass, features.Sex_male, features.Survived)
    # seaborn_plot.set_xlabel('x')
    # seaborn_plot.set_ylabel('y')
    # seaborn_plot.set_zlabel('z')
    # plt.show()

    test_df["PassengerId"] = test.PassengerId
    return features[[*final_columns, "Survived"]], test_df[["PassengerId", *final_columns]], columns


def backward_elimination(train: pd.DataFrame, test: pd.DataFrame = None) -> (pd.DataFrame, pd.DataFrame, list[str]):
    # Get dummies
    columns = ["Pclass", "Sex", "SibSp", "Fare", "Parch", "Age"]
    train_df = pd.get_dummies(train[[*columns, "Survived"]], drop_first=True)
    test_df = pd.get_dummies(test[columns], drop_first=True)
    test_df["PassengerId"] = test.PassengerId
    # Create X matrix and y vector for training
    X, y = train_df.loc[:, train_df.columns != "Survived"], train_df.Survived
    X = X.fillna(X.mean())
    # Original columns of input
    cols = list(X.columns)
    pmax = 1
    while cols:
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        # If pmax is more than .05, then we don't use the column
        if (pmax > 0.1):
            cols.remove(feature_with_p_max)
        else:
            break
    return train_df[[*cols, "Survived"]], test_df[["PassengerId", *cols]], cols


def rfe(train: pd.DataFrame, test: pd.DataFrame = None) -> (pd.DataFrame, pd.DataFrame, list[str]):
    columns = ["Pclass", "Sex", "SibSp", "Fare", "Parch", "Age"]
    train_df = pd.get_dummies(train[[*columns, "Survived"]], drop_first=True)
    test_df = pd.get_dummies(test[columns], drop_first=True)
    test_df["PassengerId"] = test.PassengerId
    # Create X matrix and y vector for training
    X, y = train_df.loc[:, train_df.columns != "Survived"], train_df.Survived
    X = X.fillna(X.mean())
    # RFE method
    nof_list = np.arange(1, len(columns))
    high_score = 0
    nof = 0
    score_list = []
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe, y_test)
        score_list.append(score)
        if (score > high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features: %d" % nof)
    print("Score with %d features: %f" % (nof, high_score))
    # Select the columns
    cols = list(X.columns)
    model = LinearRegression()
    # Initializing RFE model
    rfe = RFE(model, n_features_to_select=nof)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X, y)
    # Fitting the data to model
    model.fit(X_rfe, y)
    temp = pd.Series(rfe.support_, index=cols)
    selected_features_rfe = temp[temp == True].index
    cols = selected_features_rfe.to_list()

    return train_df[[*cols, "Survived"]], test_df[["PassengerId", *cols]], cols
