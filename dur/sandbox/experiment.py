import os
import pyodbc
import dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import datetime as dt
import optuna
import multiprocessing as mp


dotenv.load_dotenv()

# Hyperparameter tuning configuration
SEED = 42
NUM_CV_SPLITS = 10
NUM_WORKERS = min(16, mp.cpu_count())
NUM_TRIALS = 100
crossvalstrategy = StratifiedGroupKFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=SEED)
EXPERIMENT_NAME = "ModelInference-Experimentation"
STORAGE_URL = "sqlite:///optuna_lgbm.db"  # for local testing
np.random.seed(SEED)


# Load dataset
connection_string = (
    f"DRIVER={{InterSystems IRIS ODBC35}};"
    f"SERVER={os.getenv('IRIS_SERVER')};"
    f"PORT={os.getenv('IRIS_PORT')};"
    f"DATABASE={os.getenv('IRIS_NAMESPACE')};"
    f"UID={os.getenv('IRIS_USERNAME')};"
    f"PWD={os.getenv('IRIS_PASSWORD')}"
)
with pyodbc.connect(connection_string) as cnxn:
    df =  pd.read_sql(f"SELECT * FROM MLpipeline.PointSamples", cnxn)
X = df[["x", "y"]]
X["year"] = df["datetime"].dt.year
X["month"] = df["datetime"].dt.month
X["day"] = df["datetime"].dt.day
X["hour"] = df["datetime"].dt.hour
X["minute"] = df["datetime"].dt.minute
X["second"] = df["datetime"].dt.second
y = df["label"]
groups = df["datetime"].astype(str)   # optional, safe for grouping


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),  # CHANGEABLE
        "max_depth": trial.suggest_int("max_depth", 3, 50),  # CHANGEABLE
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),  # CHANGEABLE
        "num_leaves": trial.suggest_categorical("num_leaves", [16, 31, 63, 127, 255]),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),  # CHANGEABLE
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


        pipeline = Pipeline([ # TODO: ADD AUGMENTATION STEPS HERE
            ("scaler", StandardScaler()),
            ("model", lgb.LGBMClassifier(**params))
        ])

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


if __name__ == "__main__":
    
    # MLflow setup
    datetime_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    RUN_NAME = f"parent_{datetime_str}"
    STUDY_NAME = f"optuna_{datetime_str}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME, log_system_metrics=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id

        optuna.create_study(
            direction="maximize",
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
            load_if_exists=False,
        )

        mlflow.log_params({
            "n_trials": NUM_TRIALS,
            "num_workers": NUM_WORKERS,
            "cv_n_splits": crossvalstrategy.n_splits,
            "seed": SEED,
            "study_name": STUDY_NAME,
        })

        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
        )
        study.optimize(
            objective,
            n_trials=NUM_TRIALS,
            show_progress_bar=False,
            n_jobs=NUM_WORKERS,
        )
        best_params = study.best_trial.params
        best_value = study.best_value
        best_child_run_id = study.best_trial.user_attrs.get("run_id")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_f1", float(best_value))

        if best_child_run_id:
            mlflow.log_param("best_child_run_id", best_child_run_id)

        final_model = lgb.LGBMClassifier(
            **best_params,
            random_state=SEED,
            verbosity=-1,
            n_jobs=NUM_WORKERS,
        )
        final_model.fit(X, y)
        input_sample = X.sample(100, random_state=SEED)
        signature = infer_signature(input_sample, final_model.predict(input_sample))
        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            name="best_model",
            signature=signature,
            input_example=X.head(5),
        )
    os.remove("optuna_lgbm.db")