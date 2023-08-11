""" Methods to choose which are the relevant features to use in the model.
"""
import pandas as pd
from scipy.stats import pearsonr

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def filter_method(train: pd.DataFrame, test: pd.DataFrame = None) -> pd.DataFrame:
    """calculate the columns to be considered using the filter method
    The filtering here is done using correlation matrix and it is done 
    using Pearson correlation.

    Args:
        train (pd.DataFrame): train dataset

    Returns:
        pd.DataFrame: important columns
    """
    columns = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
    features = pd.get_dummies(train[[*columns, "Survived"]], drop_first=True)
    test_df = pd.get_dummies(test[columns], drop_first=True)

    # Calculate the correlations
    cor = features.corr()
    # OPTIONAL - plot correlation
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()

    # Min correlation to be considerd
    min_corr = 0.3
    # Correlations of the target column (survived)
    cor_target = abs(cor.Survived)
    # Select relevant features (more than abs(x) correlation)
    relevant_features = cor_target[cor_target > min_corr]
    final_columns = relevant_features.index.drop("Survived").to_list()

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

    return features[[*final_columns, "Survived"]], test_df[final_columns]
