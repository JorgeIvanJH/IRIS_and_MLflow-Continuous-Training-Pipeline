import os
import dotenv
import optuna
import lightgbm as lgb
import multiprocessing as mp
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import fetch_california_housing
import datetime as dt

dotenv.load_dotenv()

STORAGE_URL = "sqlite:///optuna_lgbm.db"  # for local testing


# Hyperparameter tuning configuration
NUM_WORKERS = min(16, mp.cpu_count())
NUM_TRIALS_PER_WORKER = 200
BASE_SEED = 42
NUM_CV_SPLITS = 3 # 5 or 10 would be better
EXPERIMENT_NAME = "LightGBM Hyperparameter Tuning with Optuna and MLflow"
crossvalstrategy = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=BASE_SEED)

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
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        
        crossval_score = -scores.mean()

        # Log current trial's error metric
        mlflow.log_metrics({"cv_mse_mean": crossval_score})
        for fold_idx, score in enumerate(scores):
            mlflow.log_metric(f"fold_{fold_idx}_mse", -score)

        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        return crossval_score


def run_worker(args):
    worker_id, study_name, parent_run_id = args
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id

    study = optuna.load_study(
        study_name=study_name,
        storage=STORAGE_URL,
        sampler=optuna.samplers.TPESampler(seed=BASE_SEED+worker_id),
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
    RUN_NAME = f"parent_{datetime_str}"
    STUDY_NAME = f"optuna_{datetime_str}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id


    with mlflow.start_run(run_name=RUN_NAME, log_system_metrics=True) as parent_run:
        parent_run_id = parent_run.info.run_id
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id

        optuna.create_study(
            direction="minimize",
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
            load_if_exists=False,
        )

        mlflow.log_params({
            "n_trials": NUM_TRIALS_PER_WORKER * NUM_WORKERS,
            "num_workers": NUM_WORKERS,
            "cv_n_splits": crossvalstrategy.n_splits,
            "seed": BASE_SEED,
            "study_name": STUDY_NAME,
        })

        worker_args = [
            (worker_id, STUDY_NAME, parent_run_id)
            for worker_id in range(NUM_WORKERS)
        ]
        with mp.Pool(processes=NUM_WORKERS) as pool:
            pool.map(run_worker, worker_args)

        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STORAGE_URL,
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
        input_sample = X.sample(100, random_state=BASE_SEED)
        signature = infer_signature(input_sample, final_model.predict(input_sample))
        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            name="best_model",
            signature=signature,
            input_example=X.head(5),
        )