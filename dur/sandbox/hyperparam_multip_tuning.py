# TODO: 1. name experiment and rparent run. 2. duplocate parent runs. 3. failing after 3 or more workers, 4
import os
import dotenv
import optuna
import lightgbm as lgb
import multiprocessing as mp
import mlflow
import mlflow.lightgbm
from mlflow.optuna.storage import MlflowStorage
from mlflow.models import infer_signature
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import make_scorer, mean_squared_error
import datetime as dt

dotenv.load_dotenv()

# Hyperparameter tuning configuration
NUM_WORKERS = min(2, mp.cpu_count())
NUM_TRIALS_PER_WORKER = 5
BASE_SEED = 42
NUM_CV_SPLITS = 3
EXPERIMENT_NAME = "LightGBM Hyperparameter Tuning with Optuna and MLflow"
crossvalstrategy = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=BASE_SEED)
optunasampler = optuna.samplers.TPESampler(seed=BASE_SEED)

# Load dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.columns = [col.replace(" ", "_") for col in X.columns]
y.name = "median_house_value"

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),  # CHANGEABLE
        "max_depth": trial.suggest_int("max_depth", 3, 50),  # CHANGEABLE
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),  # CHANGEABLE
        "num_leaves": trial.suggest_categorical("num_leaves", [16, 31, 63, 127, 255]),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),  # CHANGEABLE
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        "random_state": BASE_SEED,
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

        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=crossvalstrategy,
            scoring=make_scorer(mean_squared_error),
            n_jobs=1,
        )
        
        crossval_score = scores.mean()

        # Log current trial's error metric
        mlflow.log_metrics({"Cross-Validation Error": crossval_score})
        for fold_idx, score in enumerate(scores):
            mlflow.log_metric(f"Fold_{fold_idx}_Error", score)

        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        return crossval_score


def run_worker(args):
    worker_id, STUDY_NAME, mlflow_storage = args
    print("STUDY_NAME2:", STUDY_NAME)
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=mlflow_storage,
        sampler=optunasampler,
        )
    study.optimize(
        objective,
        n_trials=NUM_TRIALS_PER_WORKER,
        show_progress_bar=False,
        n_jobs=1,
    )
    return worker_id


if __name__ == "__main__":
    
    # MLflow setup
    datetime_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    STUDY_NAME = f"study_{datetime_str}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    mlflow_storage = MlflowStorage(experiment_id=experiment.experiment_id)


    with mlflow.start_run(run_name=STUDY_NAME, log_system_metrics=True) as parent_run:

        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id

        optuna.create_study(
            direction="minimize",
            sampler=optunasampler,
            study_name=STUDY_NAME,
            storage=mlflow_storage,
            load_if_exists=True,
        )

        mlflow.log_params({
            "n_trials": NUM_TRIALS_PER_WORKER * NUM_WORKERS,
            "num_workers": NUM_WORKERS,
            "cv_n_splits": crossvalstrategy.n_splits,
            "seed": BASE_SEED,
            "study_name": STUDY_NAME,
        })



        print("STUDY_NAME1:", STUDY_NAME)
        worker_args = [(worker_id, STUDY_NAME, mlflow_storage)
            for worker_id in range(NUM_WORKERS)]
        with mp.Pool(processes=NUM_WORKERS) as pool:
            pool.map(run_worker, worker_args)

        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=mlflow_storage,
        )

        best_params = study.best_trial.params
        best_value = study.best_value
        best_child_run_id = study.best_trial.user_attrs.get("run_id")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_mse", float(best_value))

        if best_child_run_id:
            mlflow.log_param("best_child_run_id", best_child_run_id)

        # Train final model on full dataset with best hyperparameters. Important: keep same seed
        final_model = lgb.LGBMRegressor(
            **best_params,
            random_state=BASE_SEED,
            verbosity=-1,
            n_jobs=1,
        )
        final_model.fit(X, y)
        signature = infer_signature(X.sample(100), final_model.predict(X.sample(100)))
        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            name="best_model",
            signature=signature,
            input_example=X.head(5),
        )