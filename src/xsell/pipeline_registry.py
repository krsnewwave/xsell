"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from xsell.pipelines import data_engineering as de
from xsell.pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_engineering_pipeline = de.create_pipeline()
    xgb_pipe = ds.create_xgb_pipeline()
    rr_pipe = ds.create_rr_pipeline()
    logres_pipe = ds.create_logres_pipeline()
    rr_ho_pipe = ds.create_rr_ho_pipeline()

    return {
        "de": data_engineering_pipeline,
        "xgb_pipe": xgb_pipe,
        "rr_pipe": rr_pipe,
        "rr_ho_pipe" : rr_ho_pipe,
        "logres_pipe": logres_pipe,
        "__default__": data_engineering_pipeline + xgb_pipe + rr_pipe + logres_pipe + rr_ho_pipe,
    }
