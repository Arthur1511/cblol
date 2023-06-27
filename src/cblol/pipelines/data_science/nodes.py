import logging
from typing import Dict, Tuple

import pandas as pd
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    X = X.drop(columns=X.select_dtypes(include=['category']).columns)
    y = data[parameters["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: Dict) -> XGBClassifier:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    classifier = XGBClassifier(**parameters)
    classifier.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    return classifier


def evaluate_model(
    classifier: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = classifier.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a AUC score of {auc_score:.2f} on test data.")
    logger.info(f"Model has a accuracy score of {acc_score:.2f} on test data.")
    logger.info(f"Model has a F-1 score of {f1:.2f} on test data.")
