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
from sklearn.metrics import plot_roc_curve
from matplotlib import pyplot as plt
from kedro_mlflow.io.metrics import MlflowMetricDataSet
import mlflow
import optuna
from functools import partial


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


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series, params: Dict) -> Dict:
    # establish early stopping validation set
    validation_split = params.pop("validation_split")
    random_state = params.pop("random_state")
    early_stopping_rounds = params.pop("early_stopping_rounds")

    # validation splits
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=random_state)

    xgb_clf = XGBClassifier(**params)

    xgb_clf.fit(X_train, y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=[(X_val, y_val)])

    dict_metrics = evaluate_model(xgb_clf, X_test, y_test)
    return {"clf": xgb_clf, "model_metrics": dict_metrics}


def rr_objective(X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              trial: optuna.trial):
    max_depth = trial.suggest_int("max_depth", 8, 64, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 50, 1000, )
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.001, 0.03, log=True)
    rr_clf = RandomForestClassifier(max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    ccp_alpha=ccp_alpha,
                                    class_weight='balanced_subsample',
                                    verbose=1)
    rr_clf.fit(X_train, y_train)
    y_proba = rr_clf.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, y_proba)
    return ap


def fit_rr_ho(X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series):
    study = optuna.create_study(direction="maximize")
    fun_rr_object = partial(rr_objective, X_train, y_train, X_test, y_test)
    # increase n_trials > 100 for better success
    study.optimize(fun_rr_object, n_trials=5)
    best_params = study.best_params

    mlflow.log_params(best_params)

    rr_clf = RandomForestClassifier(**best_params)
    rr_clf.fit(X_train, y_train)
    dict_metrics = evaluate_model(rr_clf, X_test, y_test)
    return {"clf": rr_clf, "model_metrics": dict_metrics}


def fit_rr(X_train: pd.DataFrame, y_train: pd.Series,
           X_test: pd.DataFrame, y_test: pd.Series, params: Dict) -> Dict:
    rr_clf = RandomForestClassifier(**params)
    rr_clf.fit(X_train, y_train)
    dict_metrics = evaluate_model(rr_clf, X_test, y_test)
    return {"clf": rr_clf, "model_metrics": dict_metrics}


def fit_logres(X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series, params: Dict) -> Dict:

    # scale first
    ss = StandardScaler()
    lr_clf = LogisticRegression(**params)

    lr_pipe = make_pipeline(ss, lr_clf)
    lr_pipe.fit(X_train, y_train)
    dict_metrics = evaluate_model(lr_pipe, X_test, y_test)
    return {"clf": lr_pipe, "model_metrics": dict_metrics}


def evaluate_model(clf: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates metrics.

    Args:
        clf: Trained model.
        X_test: Test X.
        y_test: Test y.
    """
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_preds = y_proba > 0.5
    ap = average_precision_score(y_test, y_proba)
    loss = log_loss(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_preds)
    logger = logging.getLogger(__name__)
    logger.info(f"Average precision: {ap}, loss: {loss}, balanced accuracy: {bal_acc}")

    return {"average_precision": {"value": ap, "step": 0},
            "loss": {"value": loss, "step": 0},
            "balanced_accuracy": {"value": bal_acc, "step": 0}}


def plot_roc(clf: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    """Plot ROC

    Args:
        clf (BaseEstimator): _description_
        X_test (pd.DataFrame): _description_
        y_test (pd.Series): _description_
    """

    # ROC curve
    plot_roc_curve(clf, X_test, y_test)
    return plt
    # plt.savefig("")
