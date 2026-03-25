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

// Placeholder for the trained model (e.g., LinearRegression from sklearn)

Property MODEL As %SYS.Python;

// Essential columns for the model to function properly (e.g., ["x", "y"] for a simple regression task)

Property ESSENTIALCOLUMNS As %List;

// Model parameters stored as a dynamic object for flexibility (e.g., {"fit_intercept": True})

Property MODELPARAMS As %DynamicObject;

// Run ID for MLflow tracking

Property MLFLOWRUNID As %String;

/*
Set default values of parameters when a new instance of AutomatedPipeline is created.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
*/
Method %OnNew(seed As %Integer = 42, testSize As %Numeric(SCALE=2) = 0.2) As %Status [ Private ]
{
    do ##class(%SYS.System).WriteToConsoleLog("Initializing MLpipeline.AutomatedPipeline instance.", 0, 0)
    set ..SEED = seed
    set ..TESTSIZE = testSize
    set ..ESSENTIALCOLUMNS = $lb("x", "y")
    set ..MODELPARAMS = {"fit_intercept": true} // Default parameters for LinearRegression

    set status = $$$OK
    set runid = ""

    try {
        set mlflow = ##class(%SYS.Python).Import("mlflow")
        do mlflow."set_tracking_uri"($System.Util.GetEnviron("MLFLOW_TRACKING_URI_IRIS"))
        do mlflow."set_experiment"(..#EXPERIMENTNAME)
        do mlflow."end_run"() // Ensure no active run before starting a new one
        set run = mlflow."start_run"()
        set runid = run.info."run_id"
        do mlflow."end_run"()
        set ..MLFLOWRUNID = runid
        do ##class(%SYS.System).WriteToConsoleLog("Created MLflow run with ID="_runid, 0, 0)
    } catch ex {
        set status = ex.AsStatus()
        do ##class(%SYS.System).WriteToConsoleLog("MLflow Error: "_ex.DisplayString(), 0, 2)
    }

    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.AutomatedPipeline with SEED="_seed_" TESTSIZE="_testSize, 0, 0)
    Quit status
}

/*
Leverages the FeatureStore to extract data and transform it to pandas DataFrame.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
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
            filters=f"datetime > '{safe_literal}'"
        )
        iris._SYS.System.WriteToConsoleLog(f"DataExtraction completed.", 0, 0)
        return df
    except Exception as e:
        print(f"DataExtraction Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"DataExtraction Error: {str(e)}", 0, 2)
        return None
}

/*
Validates the extracted DataFrame for missing values and logs the results.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
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
        
        # Data types check (e.g., ensure "x" and "y" are numeric)
        for col in essential_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                iris._SYS.System.WriteToConsoleLog(f"Column '{col}' has incorrect data type. Expected numeric.", 0, 2)

        # Significant outliers (e.g., using IQR method)
        for col in essential_cols:
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
        print(f"DataValidation Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"DataValidation Error: {str(e)}", 0, 2)
        return None
}

/* 
Prepares the validated DataFrame for model training. All non-learned transforms (renames, type casting, no .fit() transformations) should be implemented here.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
    set validDf = AP.DataValidation(df)
    set (Xtrain, Ytrain, Xtest, Ytest) = AP.DataPreparation(validDf)
*/
Method DataPreparation(df As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    try:       
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(df[["x"]], df["y"], test_size=self.TESTSIZE, random_state=self.SEED)
        iris._SYS.System.WriteToConsoleLog("DataPreparation completed.", 0, 0)
        return (Xtrain, Ytrain, Xtest, Ytest)
    except Exception as e:
        print(f"DataPreparation Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"DataPreparation Error: {str(e)}", 0, 2)
        return None
}

/*
Trains a machine learning model (e.g., LinearRegression) using the prepared data and specified parameters. Every learned transformation that requires a .fit() should be implemented here (e.g., StandardScaler, PCA, etc.).
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
    set validDf = AP.DataValidation(df)
    set result  = AP.DataPreparation(validDf)
    Set Xtrain = result."__getitem__"(0)
    Set Ytrain = result."__getitem__"(1)
    Set Xtest  = result."__getitem__"(2)
    Set Ytest  = result."__getitem__"(3)
    set model = AP.ModelTraining(Xtrain, Ytrain, {"fit_intercept": true})
*/
Method ModelTraining(Xtrain As %SYS.Python, Ytrain As %SYS.Python, params As %DynamicObject = "") As %SYS.Python [ Language = python ]
{
    import os
    import json
    import iris
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score


    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        mlflow.set_experiment(self._GetParameter("EXPERIMENTNAME"))
        mlflow.sklearn.autolog()

        # Parse Parameters
        params_dict = json.loads(params._ToJSON())
        iris._SYS.System.WriteToConsoleLog(f"ModelTraining received parameters: {params_dict}", 0, 0)

        # Define the entire pipeline with learned transformations and model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression(**params_dict))
        ])

        # Training and logging with MLflow
        with mlflow.start_run(run_id=self.MLFLOWRUNID):
            iris._SYS.System.WriteToConsoleLog("Training LinearRegression model with provided training data and parameters.", 0, 0)
            model.fit(Xtrain, Ytrain)

            # Saving Model
            model_path = os.path.join(self._GetParameter("MODELSPATH"), self.MLFLOWRUNID) # TODO: set in the initialization
            mlflow.sklearn.save_model(model, model_path) # TODO: Write a function to download and save here from the server to local based from the run id

            
            iris._SYS.System.WriteToConsoleLog("Model training completed successfully.", 0, 0)
            # Calculate metrics
            y_pred_train = model.predict(Xtrain)

            # Log manually with custom keys
            mlflow.log_metrics({
                "train_mae": mean_absolute_error(Ytrain, y_pred_train),
                "train_r2": r2_score(Ytrain, y_pred_train),
            })

            # Tag successful training for future identification of runs
            mlflow.set_tag("TrainingStatus", "OK")
            self.MODEL = model
            iris._SYS.System.WriteToConsoleLog("ModelTraining completed.", 0, 0)
            return model

    except Exception as e:
        print(f"ModelTraining Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"ModelTraining Error: {str(e)}", 0, 2)
        return None
}

/*
Evaluates the trained model using the test data and returns performance metrics (e.g., MSE, R2 Score).
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
    set validDf = AP.DataValidation(df)
    set result  = AP.DataPreparation(validDf)
    Set Xtrain = result."__getitem__"(0)
    Set Ytrain = result."__getitem__"(1)
    Set Xtest  = result."__getitem__"(2)
    Set Ytest  = result."__getitem__"(3)
    set model = AP.ModelTraining(Xtrain, Ytrain, {"fit_intercept": true})
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
    import matplotlib.pyplot as plt


    def cutom_metric(predictions, targets, metrics):
        """placeholder for a custom evaluation metric to be logged later"""
        custom_value = (predictions - targets).abs().mean() # Example: mean absolute error as a custom metric
        return MetricValue(aggregate_results={"custom_mae": custom_value})

    def custom_plot(eval_df, builtin_metrics, artifacts_dir):
        """placeholder for a custom plotting function to be logged later"""
        plt.figure(figsize=(6, 6))
        plt.xlim(0, 200)
        plt.ylim(0, 200)
        plt.scatter(eval_df["target"], eval_df["prediction"], alpha=0.5)
        plt.plot([eval_df["target"].min(), eval_df["target"].max()], [eval_df["target"].min(), eval_df["target"].max()], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Custom Prediction vs True Values Plot")
        plt.grid()
        
        # save plot to artifacts directory
        plot_path = os.path.join(artifacts_dir, "custom_plot.png")
        plt.savefig(plot_path)
        plt.close()
        return {"custom_plot": plot_path}


    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        mlflow.set_experiment(self._GetParameter("EXPERIMENTNAME"))
        mlflow.sklearn.autolog()

        custom_metric = make_metric(eval_fn=cutom_metric, greater_is_better=False, name="custom_mae")

        eval_data = Xtest.copy()
        eval_data["target"] = Ytest

        # Test logging with MLflow on the same run as training
        with mlflow.start_run(run_id=self.MLFLOWRUNID):
            signature = infer_signature(Xtest, model.predict(Xtest))
            model_info = mlflow.sklearn.log_model(model, name="model", signature=signature)
            result = mlflow.models.evaluate(
                model_info.model_uri,
                eval_data,
                targets="target",
                model_type="regressor",
                extra_metrics=[custom_metric],
                custom_artifacts=[custom_plot],
                evaluator_config={
                "log_explainer": True, # log SHAP explainer
                "explainer_type": "fast",
                "max_error_examples": 100,
                "log_model_explanations": True,
                }
            )
            mae = result.metrics["mean_absolute_error"]
            r2 = result.metrics["r2_score"]
            metrics = {"val_mae": mae, "val_r2": r2}

            # Tag successful evaluation for future identification of runs
            mlflow.set_tag("EvaluationStatus", "OK")

            iris._SYS.System.WriteToConsoleLog(f"ModelEvaluation completed. MAE: {mae}, R2 Score: {r2}", 0, 0)

            return metrics

    except Exception as e:
        print(f"ModelEvaluation Error: {str(e)}", e)
        iris._SYS.System.WriteToConsoleLog(f"ModelEvaluation Error: {str(e)}", 0, 2)
        return None
}

/*
Validation before promoting model to production. 
Compares performance to the current production model, and logs the results.
Returns the best model's run ID for promotion.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
    set validDf = AP.DataValidation(df)
    set result  = AP.DataPreparation(validDf)
    Set Xtrain = result."__getitem__"(0)
    Set Ytrain = result."__getitem__"(1)
    Set Xtest  = result."__getitem__"(2)
    Set Ytest  = result."__getitem__"(3)
    set model = AP.ModelTraining(Xtrain, Ytrain, {"fit_intercept": true})
    set newmodelmetrics = AP.ModelEvaluation(model, Xtest, Ytest)
    set bestrunid = AP.ModelValidation(newmodelmetrics, result)
*/
Method ModelValidation(newmodelmetrics As %SYS.Python, result As %SYS.Python, plotComparison As %Boolean = 1) As %SYS.Python [ Language = python ]
{
    import os
    import iris
    import mlflow
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import mean_absolute_error, r2_score
    from python_utils.utils import plot_inference, safe_model_load

    try:

        Xtrain, Ytrain, Xtest, Ytest = result

        # Get newly trained model's run information
        newmodel_run = mlflow.get_run(self.MLFLOWRUNID)

        # Get model currently in production and its run information
        gref = iris.gref("^ML.Config")
        prod_model_path = Path(gref["ModelPath"]) if gref["ModelPath"] else None

        if prod_model_path:
            prod_run_id = prod_model_path.name # file name of each saved model is the run id
            prodmodel_run = mlflow.get_run(prod_run_id)
            prod_model = safe_model_load(prod_model_path)
            iris._SYS.System.WriteToConsoleLog(f"Current model in production found and loaded. Model Path: {prod_model_path}. Run Name: {prodmodel_run.get('tags.mlflow.runName')}", 0, 0)
            # Evaluate the model currently in production on the new test set for comparison
            prod_y_pred = prod_model.predict(Xtest)
            mae = mean_absolute_error(Ytest, prod_y_pred)
            r2 = r2_score(Ytest, prod_y_pred)
            oldmodelmetrics = {"val_mae": mae, "val_r2": r2}
            iris._SYS.System.WriteToConsoleLog(f"Old Model Evaluation - MAE: {oldmodelmetrics['val_mae']}, R2 Score: {oldmodelmetrics['val_r2']}", 0, 0)
            iris._SYS.System.WriteToConsoleLog(f"New Model Evaluation - MAE: {newmodelmetrics['val_mae']}, R2 Score: {newmodelmetrics['val_r2']}", 0, 0)
            # Metrics comparison logic
            if newmodelmetrics["val_mae"] < oldmodelmetrics["val_mae"] and newmodelmetrics["val_r2"] > oldmodelmetrics["val_r2"]:
                iris._SYS.System.WriteToConsoleLog("New model outperforms the model in production.", 0, 0)
                best_run_id = self.MLFLOWRUNID
            else:
                iris._SYS.System.WriteToConsoleLog("New model does not outperform the model in production, model in production is kept.", 0, 0)
                best_run_id = prod_run_id
            if plotComparison:
                current_run = runs[runs["run_id"] == self.MLFLOWRUNID].iloc[0]
                plot_inference(self, Xtrain, Ytrain, Xtest, Ytest, prodmodel_run, newmodel_run)
        else:
            print("No previous successful run found.")
            iris._SYS.System.WriteToConsoleLog("No model found in production, best model is the newly trained one.", 0, 0)
            best_run_id = self.MLFLOWRUNID
        
        return best_run_id
    except Exception as e:
        print(f"ModelValidation Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"ModelValidation Error: {str(e)}", 0, 2)
        return None
}

/*
Runs the entire ML pipeline sequentially. Handles Each component's try/except.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set summary = AP.RunPipeline("2026-02-26 17:50:00")
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
        Xtrain, Ytrain, Xtest, Ytest = preparation_result
        # TODO: how to do k-fold cross validation here? 
        # Step 4: Model Training
        iris._SYS.System.WriteToConsoleLog("Starting ModelTraining.", 0, 0)
        model = self.ModelTraining(Xtrain, Ytrain, self.MODELPARAMS)

        # Step 5: Model Evaluation
        iris._SYS.System.WriteToConsoleLog("Starting ModelEvaluation.", 0, 0)
        newmodelmetrics = self.ModelEvaluation(model, Xtest, Ytest)

        # Step 6: Model Validation
        iris._SYS.System.WriteToConsoleLog("Starting ModelValidation.", 0, 0)
        bestrunid = self.ModelValidation(newmodelmetrics, preparation_result)

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
        print("Error during Pipeline Execution:", e)
        iris._SYS.System.WriteToConsoleLog(f"Pipeline Execution Error: {str(e)}", 0, 2)
        return {
            "MLFLOW_RUN_ID": self.MLFLOWRUNID,
            "Best_Model_Run_ID": None,
            "Model_Metrics": None,
            "Error": str(e)
        }
}

}
