"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import node, pipeline
from .nodes import split_data, fit_logres, fit_rr, fit_xgboost, evaluate_model, plot_roc, fit_rr_ho


def create_plot_roc_node():
    return node(
        func=plot_roc,
        inputs=["clf", "X_test", "y_test"],
        outputs="roc_graph",
        name="plot_roc",
    )


def create_split_node():
    return node(
        func=split_data,
        inputs=["model_input_table", "params:split_options"],
        outputs=["X_train", "X_test", "y_train", "y_test"],
        name="split_data_node",
    )


def create_xgb_pipeline(**kwargs):
    split_node = create_split_node()
    plot_node = create_plot_roc_node()

    xgb_pipe_instance = pipeline(
        [
            split_node,
            node(
                func=fit_xgboost,
                inputs=["X_train", "y_train", "X_test", "y_test",
                        "params:xgboost_params_full_feats"],
                outputs={"clf": "clf", "model_metrics": "model_metrics"},
                name="train_xgboost",
            ),
            plot_node
        ],
    )

    return pipeline(
        pipe=xgb_pipe_instance,
        inputs="model_input_table",
        namespace="xgboost_pipe",
        parameters={"params:xgboost_params_full_feats"})


def create_rr_pipeline(**kwargs):
    split_node = create_split_node()
    plot_node = create_plot_roc_node()

    rr_pipe_instance = pipeline(
        [
            split_node,
            node(
                func=fit_rr,
                inputs=["X_train", "y_train", "X_test",
                        "y_test", "params:rr_params_full_feats"],
                outputs={"clf": "clf", "model_metrics": "model_metrics"},
                name="train_rr",
            ),
            plot_node
        ]
    )

    return pipeline(
        pipe=rr_pipe_instance,
        inputs="model_input_table",
        namespace="rr_pipe",
        parameters={"params:rr_params_full_feats"})


def create_rr_ho_pipeline(**kwargs):
    split_node = create_split_node()
    plot_node = create_plot_roc_node()

    rr_pipe_instance = pipeline(
        [
            split_node,
            node(
                func=fit_rr_ho,
                inputs=["X_train", "y_train", "X_test",
                        "y_test"],
                outputs={"clf": "clf", "model_metrics": "model_metrics"},
                name="train_rr_hyperparams",
            ),
            plot_node
        ]
    )

    return pipeline(
        pipe=rr_pipe_instance,
        inputs="model_input_table",
        namespace="rr_pipe_hyperparams")


def create_logres_pipeline(**kwargs):
    split_node = create_split_node()
    plot_node = create_plot_roc_node()

    logres_pipe_instance = pipeline(
        [
            split_node,
            node(
                func=fit_logres,
                inputs=["X_train", "y_train", "X_test",
                        "y_test", "params:logres_params_full_feats"],
                outputs={"clf": "clf", "model_metrics": "model_metrics"},
                name="train_logres",
            ),
            plot_node
        ]
    )

    return pipeline(
        pipe=logres_pipe_instance,
        inputs="model_input_table",
        namespace="logres_pipe",
        parameters={"params:logres_params_full_feats"})
