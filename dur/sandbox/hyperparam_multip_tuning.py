import os
import dotenv
import optuna
import lightgbm as lgb
import multiprocessing as mp
import mlflow
import mlflow.lightgbm
from mlflow.optuna.storage import MlflowStorage
import numpy as np

from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import make_scorer, mean_squared_error
from mlflow.models import infer_signature

dotenv.load_dotenv()

NUM_WORKERS = min(3, mp.cpu_count())
NUM_TRIALS_PER_WORKER = 20
STUDY_NAME = "lightgbm_california_housing"
BASE_SEED = 42

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Hyperparameter Tuning Experiment")

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.columns = [col.replace(" ", "_") for col in X.columns]
y.name = "median_house_value"

crossvalstrategy = KFold(n_splits=3, shuffle=True, random_state=BASE_SEED)
crossvalstrategy = KFold(n_splits=3, shuffle=True, random_state=BASE_SEED)



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
        tags={"mlflow.parentRunId": parent_run_id} if parent_run_id else None,
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
        # Log current trial's error metric
        mlflow.log_metrics({"Cross-Validation Error": scores.mean()})

        for fold_idx, score in enumerate(scores):
            mlflow.log_metric(f"Fold_{fold_idx}_Error", score)

        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        return scores.mean()


def run_worker(worker_id):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Hyperparameter Tuning Experiment")

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=MlflowStorage(experiment_id=os.environ.get("MLFLOW_EXPERIMENT_ID")),
        sampler=optuna.samplers.TPESampler(seed=BASE_SEED + worker_id),
    )

    study.optimize(
        objective,
        n_trials=NUM_TRIALS_PER_WORKER,
        show_progress_bar=False,
        n_jobs=1,
    )

    return worker_id


if __name__ == "__main__":


    with mlflow.start_run(run_name="study") as parent_run:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id

        optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=BASE_SEED),
            study_name=STUDY_NAME,
            storage=MlflowStorage(experiment_id=os.environ.get("MLFLOW_EXPERIMENT_ID")),
            load_if_exists=True,
        )

        mlflow.log_params({
            "n_trials": NUM_TRIALS_PER_WORKER * NUM_WORKERS,
            "num_workers": NUM_WORKERS,
            "cv_n_splits": crossvalstrategy.n_splits,
            "seed": BASE_SEED,
            "dataset": "california_housing",
            "objective_metric": "cv_mse_mean",
            "study_name": STUDY_NAME,
        })

        with mp.Pool(processes=NUM_WORKERS) as pool:
            pool.map(run_worker, range(NUM_WORKERS))

        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage_url,
        )

        best_params = study.best_trial.params
        best_value = study.best_value
        best_child_run_id = study.best_trial.user_attrs.get("run_id")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_mse", float(best_value))

        if best_child_run_id:
            mlflow.log_param("best_child_run_id", best_child_run_id)

        final_model = lgb.LGBMRegressor(
            **best_params,
            random_state=BASE_SEED,
            verbosity=-1,
            n_jobs=1,
        )
        final_model.fit(X, y)

        signature = infer_signature(X.head(100), final_model.predict(X.head(100)))

        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            name="best_model",
            signature=signature,
            input_example=X.head(5),
        )