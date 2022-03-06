"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import node, pipeline
from .nodes import split_data, fit_logres, fit_rr, fit_xgboost, evaluate_model


def create_pipeline(**kwargs):
    split_node = node(
        func=split_data,
        inputs=["model_input_table", "params:split_options"],
        outputs=["X_train", "X_test", "y_train", "y_test"],
        name="split_data_node",
    )

    evaluate_node = node(
        func=evaluate_model,
        inputs=["clf", "X_test", "y_test"],
        outputs=None,
        name="evaluate_model_node",
    )

    xgb_pipe_instance = pipeline(
        [
            split_node,
            node(
                func=fit_xgboost,
                inputs=["X_train", "y_train", "params:xgboost_params_full_feats"],
                outputs="clf",
                name="train_xgboost",
            ),
            evaluate_node
        ],
    )
    xgb_pipe = pipeline(
        pipe=xgb_pipe_instance,
        inputs="model_input_table",
        namespace="xgboost_pipe",
        parameters={"params:xgboost_params_full_feats"})

    rr_pipe_instance = pipeline(
        [
            split_node,
            node(
                func=fit_rr,
                inputs=["X_train", "y_train", "params:rr_params_full_feats"],
                outputs="clf",
                name="train_rr",
            ),
            evaluate_node
        ]
    )

    rr_pipe = pipeline(
        pipe=rr_pipe_instance,
        inputs="model_input_table",
        namespace="rr_pipe",
        parameters={"params:rr_params_full_feats"})

    logres_pipe = pipeline(
        [
            split_node,
            node(
                func=fit_logres,
                inputs=["X_train", "y_train", "params:logres_params_full_feats"],
                outputs="clf",
                name="train_logres",
            ),
            evaluate_node
        ]
    )

    logres_pipe = pipeline(
        pipe=logres_pipe,
        inputs="model_input_table",
        namespace="logres_pipe",
        parameters={"params:logres_params_full_feats"})

    return pipeline(
        pipe=logres_pipe + rr_pipe + xgb_pipe,
        inputs="model_input_table",
        namespace="data_science",
    )
