Class MLpipeline.AutomatedPipeline Extends %RegisteredObject
{

// Experiment Name for tracking in MLflow

Parameter EXPERIMENTNAME = "AutomatedPipeline";

// Storage location for the trained models 

Parameter MODELSPATH = "/dur/models/";

// Seed for reproducibility in data splitting and model training

Property SEED As %Integer;

// Scaler object for data normalization (e.g., StandardScaler from sklearn)

Property SCALER As %SYS.Python;

// Test size for train-test split (e.g., 0.2 for 20% test data)

Property TESTSIZE As %Numeric(SCALE = 2);

// Number of Cross-Validation splits

Property NUMCVSPLITS As %Integer;

// Placeholder for the trained model (e.g., LinearRegression from sklearn)

Property MODEL As %SYS.Python;

// Essential columns for the model to function properly (e.g., ["x", "y"] for a simple regression task)

Property ESSENTIALCOLUMNS As %List;

// Ranges for hyperparameters to be used in model training and tuning (e.g., {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]})

Property MODELPARAMRANGES As %DynamicObject;

// Run ID for MLflow tracking

Property MLFLOWRUNID As %String;

// Number of Optuna trials for hyperparameter tuning

Property NUMTRIALS As %Integer;

// Number of workers for parallel processing during model training and tuning

Property NUMWORKERS As %Integer;

ClassMethod ListToDynamicArray(list As %List) As %DynamicArray
{
    Set arr = ##class(%DynamicArray).%New()
    For i=1:1:$LISTLENGTH(list) {
        Do arr.%Push($LISTGET(list,i))
    }
    Quit arr
}

/*
Set default values of parameters when a new instance of AutomatedPipeline is created.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
*/
Method %OnNew(seed As %Integer = 42, testSize As %Numeric(SCALE = 2) = 0.2, numCVSplits As %Integer = 5, numTrials As %Integer = 5, numWorkers As %Integer = 1, lrrange As %List = "", mdrange As %List = "", nerange As %List = "", nlrange As %List = "") As %Status [ Private ]
{

    Set status = $$$OK

    Try {

        Do ##class(%SYS.System).WriteToConsoleLog("Initializing MLpipeline.AutomatedPipeline instance.", 0, 0)

        Set ..SEED = seed
        Set ..TESTSIZE = testSize
        Set ..NUMCVSPLITS = numCVSplits
        Set ..NUMTRIALS = numTrials
        Set ..NUMWORKERS = numWorkers
        Set ..ESSENTIALCOLUMNS = $LISTBUILD("x", "y", "label", "datetime")

        // Default hyperparameter ranges for model retraining
        If (lrrange = "") { Set lrrange = $LISTBUILD(0.001, 0.2) }
        If (mdrange = "") { Set mdrange = $LISTBUILD(3, 50) }
        If (nerange = "") { Set nerange = $LISTBUILD(50, 1000) }
        If (nlrange = "") { Set nlrange = $LISTBUILD(16, 31, 63, 127, 255) }
        Set ranges = ##class(%DynamicObject).%New()
        Do ranges.%Set("learning_rate", ..ListToDynamicArray(lrrange))
        Do ranges.%Set("max_depth", ..ListToDynamicArray(mdrange))
        Do ranges.%Set("n_estimators", ..ListToDynamicArray(nerange))
        Do ranges.%Set("num_leaves", ..ListToDynamicArray(nlrange))
        Do ranges.%Set("max_bin", ..ListToDynamicArray($LISTBUILD(63,127,255)))
        Do ranges.%Set("random_state", ..SEED)
        Do ranges.%Set("verbosity", -1)
        Do ranges.%Set("n_jobs", 1) // Warning: avoid nested parallelism among Optuna trials, cross-validation, and model training. We use NUMWORKERS only for Optuna trials
        Set ..MODELPARAMRANGES = ranges

        // MLflow setup
        Set runid = ""
        Set mlflow = ##class(%SYS.Python).Import("mlflow")
        Do mlflow."set_tracking_uri"($SYSTEM.Util.GetEnviron("MLFLOW_TRACKING_URI_IRIS"))
        Do mlflow."set_experiment"(..#EXPERIMENTNAME)
        Do mlflow."end_run"() // Ensure no active run before starting a new one
        Set run = mlflow."start_run"()
        Set runid = run.info."run_id"
        Do mlflow."end_run"()
        Set ..MLFLOWRUNID = runid
        Do ##class(%SYS.System).WriteToConsoleLog("Created MLflow run with ID="_runid, 0, 0)
        Do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.AutomatedPipeline with SEED="_seed_" TESTSIZE="_testSize_" NUMCVSPLITS="_numCVSplits_" NUMTRIALS="_numTrials, 0, 0)
    } Catch ex {
        Set status = ex.AsStatus()
        Do ##class(%SYS.System).WriteToConsoleLog("MLflow Error: "_ex.DisplayString(), 0, 2)
    }
    Quit status
}

/*
Leverages the FeatureStore to extract data and transform it to pandas DataFrame.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set df = AP.DataExtraction("2026-05-14 00:00:00")
*/
Method DataExtraction(datetimestr As %String) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    import datetime as dt
    from python_utils.utils import IRIS_DBQuery
    
    try:
        parsed_dt = dt.datetime.strptime(datetimestr, "%Y-%m-%d %H:%M:%S")
        safe_literal = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
        df = IRIS_DBQuery(
            schema="MLpipeline", 
            tablename="PointSamples", 
            columns="*", 
            filters=f"datetime >= '{safe_literal}'"
        )
        iris._SYS.System.WriteToConsoleLog(f"DataExtraction completed.", 0, 0)
        return df
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"DataExtraction Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"DataExtraction Error: {msg}", 0, 2)
        return None
}

/*
Validates the extracted DataFrame for missing values and logs the results.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set df = AP.DataExtraction("2026-05-14 00:00:00")
    set validDf = AP.DataValidation(df)
*/
Method DataValidation(df As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    
    try:
        # Missing Values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            iris._SYS.System.WriteToConsoleLog(f"Missing values in DataFrame: {missing_values}", 0, 1)

        # Missing Required Columns
        essential_cols = iris.cls("%SYS.Python").ToList(self.ESSENTIALCOLUMNS)
        missing_columns = [col for col in essential_cols if col not in df.columns]
        if missing_columns:
            iris._SYS.System.WriteToConsoleLog(f"Missing essential columns in DataFrame: {missing_columns}", 0, 2)
            return None
        
        # Data types checks
        for col in essential_cols:
            # Numeric columns
            if col in ["x", "y"] and not pd.api.types.is_numeric_dtype(df[col]):
                iris._SYS.System.WriteToConsoleLog(f"Column '{col}' has incorrect data type. Expected numeric.", 0, 2)
            # Character columns
            if col == "label" and not pd.api.types.is_object_dtype(df[col]):
                iris._SYS.System.WriteToConsoleLog(f"Column '{col}' has incorrect data type. Expected character.", 0, 2)
            # Datetime columns
            if col == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
                iris._SYS.System.WriteToConsoleLog(f"Column '{col}' has incorrect data type. Expected datetime.", 0, 2)

        # Significant outliers (e.g., using IQR method)
        for col in essential_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            df_col = df[col]
            Q1 = df_col.quantile(0.25)
            Q3 = df_col.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df_col < (Q1 - 1.5 * IQR)) | (df_col > (Q3 + 1.5 * IQR))).sum()
            if outliers.sum() > 0:
                iris._SYS.System.WriteToConsoleLog(f"Outliers detected in DataFrame: {outliers}", 0, 1)

        # End of validation
        iris._SYS.System.WriteToConsoleLog("DataValidation completed.", 0, 0)
        return df
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"DataValidation Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"DataValidation Error: {msg}", 0, 2)
        return None
}

/* 
Prepares the validated DataFrame for model training. All non-learned transforms (renames, type casting, no .fit() transformations) should be implemented here.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set df = AP.DataExtraction("2026-05-14 00:00:00")
    set validDf = AP.DataValidation(df)
    set result = AP.DataPreparation(validDf)
    set crossvalidator = result."__getitem__"(0)
    set Xtrainval = result."__getitem__"(1)
    set Ytrainval = result."__getitem__"(2)
    set Xtest = result."__getitem__"(3)
    set Ytest = result."__getitem__"(4)
*/
Method DataPreparation(df As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score, GroupShuffleSplit
    
    
    try:
        # Feature engineering
        X = df[["x", "y"]].copy()
        X["year"] = df["datetime"].dt.year
        X["month"] = df["datetime"].dt.month
        X["day"] = df["datetime"].dt.day
        X["hour"] = df["datetime"].dt.hour
        X["minute"] = df["datetime"].dt.minute
        X["second"] = df["datetime"].dt.second
        y = df["label"]
        groups = df["datetime"].astype(str)

        
        # Group-aware train/test split
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.TESTSIZE,random_state=self.SEED)
        trainval_idx, test_idx = next(splitter.split(X, y, groups=groups))
        Xtrainval = X.iloc[trainval_idx]
        Xtest = X.iloc[test_idx]
        Ytrainval = y.iloc[trainval_idx]
        Ytest = y.iloc[test_idx]
        groups_trainval = groups.iloc[trainval_idx]

        # Group-aware stratified CV inside trainval only
        crossvalstrategy = StratifiedGroupKFold(
            n_splits=self.NUMCVSPLITS, 
            shuffle=True, 
            random_state=self.SEED)

        crossvalidator = lambda pipeline: cross_val_score(
                pipeline,
                Xtrainval,
                Ytrainval,
                cv=crossvalstrategy,
                scoring="f1_macro",
                groups=groups_trainval,
                n_jobs=1,) # Warning: avoid nested parallelism among Optuna trials, cross-validation, and model training. We use n_jobs > 1 only for Optuna trials

        iris._SYS.System.WriteToConsoleLog("Method for Cross-Validation completed.", 0, 0)
        return (crossvalidator, Xtrainval, Ytrainval, Xtest, Ytest)
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"DataPreparation Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"DataPreparation Error: {msg}", 0, 2)
        return None
}

/*
Trains a machine learning model (e.g., LinearRegression) using the prepared data and specified parameters. Every learned transformation that requires a .fit() should be implemented here (e.g., StandardScaler, PCA, etc.).
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set df = AP.DataExtraction("2026-05-14 00:00:00")
    set validDf = AP.DataValidation(df)
    set result = AP.DataPreparation(validDf)
    set crossvalidator = result."__getitem__"(0)
    set Xtrainval = result."__getitem__"(1)
    set Ytrainval = result."__getitem__"(2)
    set Xtest = result."__getitem__"(3)
    set Ytest = result."__getitem__"(4)
    set model = AP.ModelTraining(Xtrainval, Ytrainval, crossvalidator)
*/
Method ModelTraining(Xtrainval As %SYS.Python, Ytrainval As %SYS.Python, crossvalidator As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import os
    import json
    import iris
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    import datetime as dt
    import optuna
    import lightgbm as lgb
    from urllib.parse import quote_plus
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    def parse_os_dict(os_dict):
        json_text = os_dict._ToJSON()
        p_dict = json.loads(json_text)
        return p_dict

    class Objective:
        """
        Objective class for Optuna hyperparameter optimization, integrated with MLflow for logging each trial's parameters, metrics, and artifacts.
        
        Args:
            pipeline: A function that takes hyperparameters as input and returns a scikit-learn Pipeline object.
            paramranges: A dictionary containing the ranges for each hyperparameter to be optimized.
            crossvalidator: A function that takes a pipeline as input and returns cross-validation scores.
        
        Returns:
            The mean cross-validation score for the given set of hyperparameters.
        """

        def __init__(self, pipeline, paramranges, crossvalidator):
            self.pipeline = pipeline
            self.paramranges = paramranges
            self.crossvalidator = crossvalidator

        def __call__(self, trial):

            params = {
                "learning_rate": trial.suggest_float("learning_rate", *self.paramranges["learning_rate"], log=True),
                "max_depth": trial.suggest_int("max_depth", *self.paramranges["max_depth"]),
                "n_estimators": trial.suggest_int("n_estimators", *self.paramranges["n_estimators"]),
                "num_leaves": trial.suggest_categorical("num_leaves", self.paramranges["num_leaves"]),
                "max_bin": trial.suggest_categorical("max_bin", self.paramranges["max_bin"]),
                "random_state": self.paramranges["random_state"],
                "verbosity": self.paramranges["verbosity"],
                "n_jobs": self.paramranges["n_jobs"],
            }
            # Get MLflow parent run ID for nested logging
            MLflowParentRun = iris.gref("^MLflowParentRun")
            parent_run_id = MLflowParentRun["RunID"]
            # Logging child runs for each Optuna trial under the same parent run
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True, parent_run_id=parent_run_id) as child_run:
                scores = self.crossvalidator(self.pipeline(params))
                crossval_score = scores.mean()
                # Logging metrics
                for fold_idx, score in enumerate(scores):
                    mlflow.log_metric(f"fold_{fold_idx}_f1", score)
                mlflow.log_metrics({"cv_f1_mean": crossval_score})
                trial.set_user_attr("run_id", child_run.info.run_id)
                return crossval_score


    try:

        # Model training Pipeline definition
        pipeline = lambda params: Pipeline([ # TODO: ADD AUGMENTATION STEPS HERE
            ("scaler", StandardScaler()),
            ("model", lgb.LGBMClassifier(**params))
        ]) 
        iris._SYS.System.WriteToConsoleLog(f"Model training pipeline defined as: {pipeline}", 0, 0)

        # MLflow setup
        datetime_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        RUN_NAME = f"parent_{datetime_str}"
        STUDY_NAME = f"optuna_{datetime_str}"
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        mlflow.set_experiment(self._GetParameter("EXPERIMENTNAME"))
        #; mlflow.sklearn.autolog()


        with mlflow.start_run(run_id=self.MLFLOWRUNID, run_name=RUN_NAME, log_system_metrics=True) as parent_run:
            MLflowParentRun = iris.gref("^MLflowParentRun")
            MLflowParentRun["RunID"] = parent_run.info.run_id
            mlflow.log_params({
                "n_trials": self.NUMTRIALS,
                "num_workers": self.NUMWORKERS,
                "cv_n_splits": self.NUMCVSPLITS,
                "seed": self.SEED,
                "study_name": STUDY_NAME,
            })

            study = optuna.create_study(
                direction="maximize",
                study_name=STUDY_NAME,
                storage=optuna.storages.InMemoryStorage(),
                load_if_exists=False,
            )
            # Parse Parameters
            paramsranges = parse_os_dict(self.MODELPARAMRANGES)
            iris._SYS.System.WriteToConsoleLog(f"ModelTraining received parameter ranges: {paramsranges}", 0, 0)

            # Run Optuna optimization
            study.optimize(Objective(pipeline, paramsranges, crossvalidator),
                n_trials=self.NUMTRIALS,
                n_jobs=self.NUMWORKERS
                )
            iris._SYS.System.WriteToConsoleLog(f"Optuna optimization completed. Best trial: {study.best_trial.number} with value: {study.best_value} and parameters: {study.best_trial.params}", 0, 0)

            # Logging best trial's parameters and metrics to MLflow    
            best_params = study.best_trial.params
            best_value = study.best_value
            best_child_run_id = study.best_trial.user_attrs.get("run_id")
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_cv_f1", float(best_value))
            if best_child_run_id:
                mlflow.log_param("best_child_run_id", best_child_run_id)

            # Save the best model from the best trial
            model_path = os.path.join(self._GetParameter("MODELSPATH"), self.MLFLOWRUNID)
            final_model = lgb.LGBMClassifier(
                **best_params,
                random_state=self.SEED,
                verbosity=-1,
                n_jobs=1,
            )
            final_model.fit(Xtrainval, Ytrainval)
            input_sample = Xtrainval.sample(100, random_state=self.SEED)
            signature = infer_signature(input_sample, final_model.predict(input_sample))
            mlflow.lightgbm.log_model(
                lgb_model=final_model,
                name="best_model",
                signature=signature,
                input_example=Xtrainval.head(5),
            )
            mlflow.lightgbm.save_model(
                final_model, 
                model_path,
                signature=signature,
                metadata={"run_id": parent_run.info.run_id},
                )
            iris._SYS.System.WriteToConsoleLog("Model training completed successfully.", 0, 0)

            # TODO: Log dataset used for training
            #; trainval_data = Xtrainval.copy()
            #; trainval_data["label"] = Ytrainval
            #; mlflow.log_input(trainval_data, context="training")

            # Tag successful training for future identification of runs
            mlflow.set_tag("TrainingStatus", "OK")
            self.MODEL = final_model
            iris._SYS.System.WriteToConsoleLog("ModelTraining completed.", 0, 0)
            return final_model

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"ModelTraining Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"ModelTraining Error: {msg}", 0, 2)
        return None
}

/*
Evaluates the trained model using the test data and returns performance metrics (e.g., F1 score, MAE, R2, etc.). Also creates SHAP explanations and logs them as artifacts in MLflow along with custom metrics and plots.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set df = AP.DataExtraction("2026-05-14 00:00:00")
    set validDf = AP.DataValidation(df)
    set result = AP.DataPreparation(validDf)
    set crossvalidator = result."__getitem__"(0)
    set Xtrainval = result."__getitem__"(1)
    set Ytrainval = result."__getitem__"(2)
    set Xtest = result."__getitem__"(3)
    set Ytest = result."__getitem__"(4)
    set model = AP.ModelTraining(Xtrainval, Ytrainval, crossvalidator)
    set newmodelmetrics = AP.ModelEvaluation(model, Xtest, Ytest)
*/
Method ModelEvaluation(model As %SYS.Python, Xtest As %SYS.Python, Ytest As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import os
    import iris
    import pandas as pd
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.models import make_metric
    from mlflow.metrics.base import MetricValue
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt


    def custom_metric(predictions, targets, metrics):
        """
        Example of a custom metric.
        This is a placeholder for a custom evaluation metric to be logged later.

        Here you can put any custom logic to compute a metric based on the predictions and targets. E.g. ROI, profit, or any business-specific metric that is not covered by standard metrics. TODO: Emphasize this in the documentation and presentation
        """
        custom_value = (f1_score(targets, predictions, average="weighted") * 3.1416 )  # Just an example using F1 score multiplied by a random factor to simulate a custom metric. Replace with actual logic.
        return MetricValue(aggregate_results={"custom_metric": custom_value})

    def custom_plot(eval_df, builtin_metrics, artifacts_dir):
        """
        Custom plotting function logged as an MLflow artifact.
        We need local reference to Xtest for the x and y coordinates because eval_df only contains the predictions and targets when passed to MLflow.
        """
        color_map = {'a': 'blue', 'b': 'yellow', 'c': 'green', 'd': 'red'}
        plot_df = eval_df.copy().reset_index(drop=True)
        # Reattach coordinates from the original test data.
        coords = Xtest[["x", "y"]].copy().reset_index(drop=True)
        plot_df["x"] = coords["x"]
        plot_df["y"] = coords["y"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for label, color in color_map.items():
            actual = plot_df[plot_df["target"] == label]
            if not actual.empty:
                axes[0].scatter(actual.get("x"), actual.get("y"), s=12, c=color, label=label, alpha=0.7)

            predicted = plot_df[plot_df["prediction"] == label]
            if not predicted.empty:
                axes[1].scatter(predicted.get("x"), predicted.get("y"), s=12, c=color, label=label, alpha=0.7)

        axes[0].set_title("Actual labels")
        axes[1].set_title("Model predictions")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()

        plt.tight_layout()

        # save plot to artifacts directory
        plot_path = os.path.join(artifacts_dir, "custom_plot.png")
        plt.savefig(plot_path)
        plt.close()
        return {"custom_plot": plot_path}


    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        mlflow.set_experiment(self._GetParameter("EXPERIMENTNAME"))
        #; mlflow.lightgbm.autolog()

        custom_metric = make_metric(eval_fn=custom_metric, greater_is_better=True, name="custom_metric")

        eval_data = Xtest.copy()
        eval_data["label"] = Ytest

        # Test logging with MLflow on the same run as training
        with mlflow.start_run(run_id=self.MLFLOWRUNID):
            signature = infer_signature(Xtest, model.predict(Xtest))
            model_info = mlflow.lightgbm.log_model(model, name="model", signature=signature)
            result = mlflow.models.evaluate(
                model_info.model_uri,
                eval_data,
                targets="label",
                model_type="classifier",
                extra_metrics=[custom_metric],
                custom_artifacts=[custom_plot],
                evaluator_config={
                "log_explainer": True, # log SHAP explainer
                "explainer_type": "fast",
                "max_error_examples": 100,
                "log_model_explanations": True,
                }
            )
            f1 = result.metrics["f1_score"]
            metrics = {
                "val_f1": f1,
            }

            # Tag successful evaluation for future identification of runs
            mlflow.set_tag("EvaluationStatus", "OK")

            iris._SYS.System.WriteToConsoleLog(f"ModelEvaluation completed. F1 Score: {f1}", 0, 0)

            return metrics

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"ModelEvaluation Error: {msg}", e)
        iris._SYS.System.WriteToConsoleLog(f"ModelEvaluation Error: {msg}", 0, 2)
        return None
}

/*
Validation before promoting model to production. 
Compares performance to the current production model, and logs the results.
Returns the best model's run ID for promotion.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set df = AP.DataExtraction("2026-05-14 00:00:00")
    set validDf = AP.DataValidation(df)
    set result = AP.DataPreparation(validDf)
    set crossvalidator = result."__getitem__"(0)
    set Xtrainval = result."__getitem__"(1)
    set Ytrainval = result."__getitem__"(2)
    set Xtest = result."__getitem__"(3)
    set Ytest = result."__getitem__"(4)
    set model = AP.ModelTraining(Xtrainval, Ytrainval, crossvalidator)
    set newmodelmetrics = AP.ModelEvaluation(model, Xtest, Ytest)
    set bestrunid = AP.ModelValidation(newmodelmetrics, Xtest, Ytest)
*/
Method ModelValidation(newmodelmetrics As %SYS.Python, Xtest As %SYS.Python, Ytest As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import os
    import iris
    import mlflow
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import f1_score
    from python_utils.utils import safe_model_load

    try:

        # Get newly trained model's run information
        newmodel_run = mlflow.get_run(self.MLFLOWRUNID)

        # Get model currently in production and its run information
        gref = iris.gref("^ML.Config")
        prod_model_path = Path(gref["ModelPath"]) if gref["ModelPath"] else None

        if prod_model_path:
            prod_run_id = prod_model_path.name # file name of each saved model is the run id
            prodmodel_run = mlflow.get_run(prod_run_id)
            prod_model = safe_model_load(prod_model_path)
            iris._SYS.System.WriteToConsoleLog(f"Current model in production found and loaded. Model Path: {prod_model_path}. Run Name: {prodmodel_run.data.tags.get('mlflow.runName')}", 0, 0)
            # Evaluate the model currently in production on the new test set for comparison
            prod_y_pred = prod_model.predict(Xtest)
            f1 = f1_score(Ytest, prod_y_pred, average="weighted")
            oldmodelmetrics = {"val_f1": f1}
            iris._SYS.System.WriteToConsoleLog(f"Old Model Evaluation - F1 Score: {oldmodelmetrics['val_f1']}", 0, 0)
            iris._SYS.System.WriteToConsoleLog(f"New Model Evaluation - F1 Score: {newmodelmetrics['val_f1']}", 0, 0)
            # Metrics comparison logic
            if newmodelmetrics["val_f1"] > oldmodelmetrics["val_f1"]:
                iris._SYS.System.WriteToConsoleLog("New model outperforms the model in production.", 0, 0)
                best_run_id = self.MLFLOWRUNID
            else:
                iris._SYS.System.WriteToConsoleLog("New model does not outperform the model in production, model in production is kept.", 0, 0)
                best_run_id = prod_run_id
        else:
            print("No previous successful run found.")
            iris._SYS.System.WriteToConsoleLog("No model found in production, best model is the newly trained one.", 0, 0)
            best_run_id = self.MLFLOWRUNID
        
        return best_run_id
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"ModelValidation Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"ModelValidation Error: {msg}", 0, 2)
        return None
}

/*
Runs the entire ML pipeline sequentially. Handles Each component's try/except.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New(42, 0.2, 5, 5, 16)
    set summary = AP.RunPipeline("2026-05-14 00:00:00")
*/
Method RunPipeline(datetimestr As %String) As %SYS.Python [ Language = python ]
{
    import os
    import iris
    import pandas as pd
    import mlflow
    
    try:
        iris._SYS.System.WriteToConsoleLog("Starting Automated ML Pipeline execution.", 0, 0)
        
        # Step 1: Data Extraction
        iris._SYS.System.WriteToConsoleLog(f"Starting DataExtraction for date: {datetimestr}", 0, 0)
        df = self.DataExtraction(datetimestr)
        
        # Step 2: Data Validation
        iris._SYS.System.WriteToConsoleLog("Starting DataValidation.", 0, 0)
        valid_df = self.DataValidation(df)

        # Step 3: Data Preparation
        iris._SYS.System.WriteToConsoleLog("Starting DataPreparation.", 0, 0)
        preparation_result = self.DataPreparation(valid_df)
        crossvalidator, Xtrainval, Ytrainval, Xtest, Ytest = preparation_result

        # Step 4: Model Training
        iris._SYS.System.WriteToConsoleLog("Starting ModelTraining.", 0, 0)
        model = self.ModelTraining(Xtrainval, Ytrainval, crossvalidator)

        # Step 5: Model Evaluation
        iris._SYS.System.WriteToConsoleLog("Starting ModelEvaluation.", 0, 0)
        newmodelmetrics = self.ModelEvaluation(model, Xtest, Ytest)

        # Step 6: Model Validation
        iris._SYS.System.WriteToConsoleLog("Starting ModelValidation.", 0, 0)
        bestrunid = self.ModelValidation(newmodelmetrics, Xtest, Ytest)

        # Model Promoting process
        iris._SYS.System.WriteToConsoleLog("Starting PromotingLogic.", 0, 0)
        iris.MLpipeline.ModelServing.PromotingLogic(self.MLFLOWRUNID, bestrunid)

        iris._SYS.System.WriteToConsoleLog("Automated ML Pipeline execution completed.", 0, 0)
        return {
            "MLFLOW_RUN_ID": self.MLFLOWRUNID,
            "Best_Model_Run_ID": bestrunid,
            "Model_Metrics": newmodelmetrics
        }

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"RunPipeline Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"RunPipeline Error: {msg}", 0, 2)
        return {
            "MLFLOW_RUN_ID": self.MLFLOWRUNID,
            "Best_Model_Run_ID": None,
            "Model_Metrics": None,
            "Error": str(e)
        }
}

}
