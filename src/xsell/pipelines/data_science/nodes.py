"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

from distutils.log import Log
import logging
from typing import Any, Dict, Tuple

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data.drop(columns=["Response"])
    y = data["Response"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: Dict) -> XGBClassifier:
    # establish early stopping validation set
    validation_split = params.pop("validation_split")
    random_state = params.pop("random_state")
    early_stopping_rounds = params.pop("early_stopping_rounds")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=random_state)

    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(X_train, y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=[(X_val, y_val)])
    return xgb_clf


def fit_rr(X_train: pd.DataFrame, y_train: pd.Series, params: Dict) -> RandomForestClassifier:
    rr_clf = RandomForestClassifier(**params)
    rr_clf.fit(X_train, y_train)
    return rr_clf


def fit_logres(X_train: pd.DataFrame, y_train: pd.Series, params: Dict) -> LogisticRegression:
    # scale first
    ss = StandardScaler()
    lr_clf = LogisticRegression(**params)

    lr_pipe = make_pipeline(ss, lr_clf)
    lr_pipe.fit(X_train, y_train)
    return lr_pipe


def evaluate_model(clf: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_preds = y_proba > 0.5
    ap = average_precision_score(y_test, y_proba)
    loss = log_loss(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_preds)
    logger = logging.getLogger(__name__)
    logger.info(f"Average precision: {ap}, loss: {loss}, balanced accuracy: {bal_acc}")
