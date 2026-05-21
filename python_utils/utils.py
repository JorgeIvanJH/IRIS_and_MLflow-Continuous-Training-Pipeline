import os
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import pandas as pd
import time
import iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
import lightgbm as lgb

SEED = 42  # TODO: Extract from AutomatedPipeline.cls
NUM_CV_SPLITS = 10  # TODO: Extract from AutomatedPipeline.cls
crossvalstrategy = StratifiedGroupKFold(
    n_splits=NUM_CV_SPLITS, shuffle=True, random_state=SEED
)


def measure_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    return wrapper


def save_mlflow_model(runid: str):
    """
    Loads a model from MLflow using the provided run ID and re-saves it to the path specified in the MODELSPATH parameter.
    """
    import iris
    import mlflow
    import os
    import dotenv

    dotenv.load_dotenv()

    try:
        iris._SYS.System.WriteToConsoleLog(
            f"Attempting to re-save model for Run ID: {runid}", 0, 0
        )
        # Use the internal Docker network URL for the MLflow container
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        model_uri = f"runs:/{runid}/model"
        model = mlflow.lightgbm.load_model(model_uri)
        base_path = iris.cls("MLpipeline.AutomatedPipeline")._GetParameter("MODELSPATH")
        model_path = os.path.join(base_path, runid)
        mlflow.lightgbm.save_model(model, path=model_path)
        iris._SYS.System.WriteToConsoleLog(f"Model re-saved to: {model_path}", 0, 0)
        return True
    except Exception as e:
        print(f"ReSaveMLflowModel Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"ReSaveMLflowModel Error: {str(e)}", 0, 2)
        return False


def safe_model_load(model_path: str):
    """
    Safely loads a model from the specified path. If loading fails, it attempts to re-save the model and load it again.
    """
    import iris
    import mlflow
    import os

    try:
        model = mlflow.lightgbm.load_model(model_path)
        model.mlflow_model_info = mlflow.models.get_model_info(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(
            f"Error loading model from {model_path}: {str(e)}", 0, 2
        )
        # Extract run ID from the model path and attempt to re-save the model
        run_id = os.path.basename(model_path)
        print(f"Attempting to re-save model for Run ID: {run_id}")
        iris._SYS.System.WriteToConsoleLog(
            f"Attempting to re-save model for Run ID: {run_id}", 0, 0
        )
        if save_mlflow_model(run_id):
            try:
                print(
                    f"Attempting to load model again from {model_path} after re-saving."
                )
                iris._SYS.System.WriteToConsoleLog(
                    f"Attempting to load model again from {model_path} after re-saving.",
                    0,
                    0,
                )
                model = mlflow.lightgbm.load_model(model_path)
                model.mlflow_model_info = mlflow.models.get_model_info(model_path)
                return model
            except Exception as e:
                print(
                    f"Error loading model after re-saving from {model_path}: {str(e)}"
                )
                iris._SYS.System.WriteToConsoleLog(
                    f"Error loading model after re-saving from {model_path}: {str(e)}",
                    0,
                    2,
                )
                return None
        else:
            return None


def IRIS_DBQuery(
    schema: str, tablename: str, columns: str = "*", filters: str = ""
) -> pd.DataFrame:
    """
    Executes a database query against an IRIS database and returns the results as a pandas DataFrame.
    Args:
        schema (str): The database schema to query.
        tablename (str): The table name to query.
        columns (str): The columns to select (default is "*").
        filters (str): Optional SQL filters to apply to the query. Ignoring WHERE clause (e.g. datetime > '2023-01-01').
    Returns:
        pd.DataFrame: The query results as a pandas DataFrame.
    """
    import iris
    import pandas as pd

    # Basic Identifier Validation (Prevents SQL injection)
    if not (schema.isalnum() and tablename.replace("_", "").isalnum()):
        raise ValueError("Invalid Schema or Table name.")

    try:
        FS = iris.MLpipeline.FeatureStore._New()
        os_rs = FS.DataExtraction(schema, tablename, columns, filters)
        py_rs = iris.cls("%SYS.Python.SQLResultSet")._New(os_rs)
        df = py_rs.dataframe()
        if df.empty:
            iris._SYS.System.WriteToConsoleLog(
                "IRIS_DBQuery returned empty result set.", 0, 1
            )
        return df
    except Exception as e:
        print(f"IRIS_DBQuery Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"IRIS_DBQuery Error: {str(e)}", 0, 2)
        return pd.DataFrame()


class Objective:
    def __init__(self, X, y, groups, pipeline, paramranges, crossvalidator):
        self.X = X
        self.y = y
        self.groups = groups
        self.pipeline = pipeline
        self.paramranges = paramranges
        self.crossvalidator = crossvalidator

    def __call__(self, trial):

        parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")  # TODO:

        with mlflow.start_run(
            run_name=f"trial_{trial.number}", nested=True, parent_run_id=parent_run_id
        ) as child_run:

            scores = self.crossvalidator(self.pipeline)

            crossval_score = scores.mean()

            # Log current trial's error metric
            mlflow.log_metrics({"cv_f1_mean": crossval_score})
            for fold_idx, score in enumerate(scores):
                mlflow.log_metric(f"fold_{fold_idx}_f1", score)

            # Make it easy to retrieve the best-performing child run later
            trial.set_user_attr("run_id", child_run.info.run_id)

            return crossval_score


def objective(trial):
    params = {  # TODO: Define closer to AutomatedPipeline.cls and make sure to log them as parameters in MLflow for better tracking
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.001, 0.2, log=True
        ),  # CHANGEABLE
        "max_depth": trial.suggest_int("max_depth", 3, 50),  # CHANGEABLE
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),  # CHANGEABLE
        "num_leaves": trial.suggest_categorical("num_leaves", [16, 31, 63, 127, 255]),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 1e-8, 10.0, log=True
        ),  # CHANGEABLE
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        "random_state": SEED,
        "verbosity": -1,
        "n_jobs": 1,
    }

    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")

    with mlflow.start_run(
        run_name=f"trial_{trial.number}",
        nested=True,
        parent_run_id=parent_run_id,
        # tags={"mlflow.parentRunId": parent_run_id} if parent_run_id else None,
    ) as child_run:

        pipeline = Pipeline(
            [  # TODO: ADD AUGMENTATION STEPS HERE
                ("scaler", StandardScaler()),
                ("model", lgb.LGBMClassifier(**params)),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=crossvalstrategy,
            scoring="f1_macro",
            groups=groups,  # Use datetime as the grouping variable to prevent data leakage
            n_jobs=1,
        )

        crossval_score = scores.mean()

        # Log current trial's error metric
        mlflow.log_metrics({"cv_f1_mean": crossval_score})
        for fold_idx, score in enumerate(scores):
            mlflow.log_metric(f"fold_{fold_idx}_f1", score)

        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        return crossval_score
