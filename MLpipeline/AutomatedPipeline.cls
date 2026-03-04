Class MLpipeline.AutomatedPipeline Extends %RegisteredObject
{

// Experiment Name for tracking purposes in MLflow or similar platforms (CONSTANT)

Parameter EXPERIMENTNAME = "PRODUCTION_CT_Pipeline";

// Seed for reproducibility in data splitting and model training

Property SEED As %Integer;

// Scaler object for data normalization (e.g., StandardScaler from sklearn)

Property SCALER As %SYS.Python;

// Test size for train-test split (e.g., 0.2 for 20% test data)

Property TESTSIZE As %Numeric(SCALE = 2);

// Placeholder for the trained model (e.g., LinearRegression from sklearn)

Property MODEL As %SYS.Python;

// Model parameters stored as a dynamic object for flexibility (e.g., {"fit_intercept": True})

Property MODELPARAMS As %DynamicObject;

// Run ID for MLflow tracking

Property MLFLOWRUNID As %String;

/*
Set default values of parameters when a new instance of AutomatedPipeline is created.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
*/
Method %OnNew(seed As %Integer = 42, testSize As %Numeric(SCALE=2) = 0.2, modelParams As %DynamicObject = "") As %Status [ Private ]
{
    set ..SEED = seed
    set ..TESTSIZE = testSize
    if modelParams = "" {
        set ..MODELPARAMS = {"fit_intercept": true} // Default parameters for LinearRegression
    } else {
        set ..MODELPARAMS = modelParams
    }

    try {
        set mlflow = ##class(%SYS.Python).Import("mlflow")
        do mlflow."end_run"() // Ensure no active run before starting a new one
        set run = mlflow."start_run"()
        set runid = run.info."run_id"
        do mlflow."end_run"()
        set ..MLFLOWRUNID = runid
        do ##class(%SYS.System).WriteToConsoleLog("Created MLflow run  with ID=" _ runid, 0, 0)
        set status = $$$OK
    } catch ex {
        set status = ex.AsStatus()
        do ##class(%SYS.System).WriteToConsoleLog("MLflow Error: " _ ex.DisplayString(), 0, 2)
    }
    set ..MLFLOWRUNID = runid
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.AutomatedPipeline with SEED=" _ seed _ " and TESTSIZE=" _ testSize _ " and MODELPARAMS=" _ ..MODELPARAMS, 0, 0)
    Quit $$$OK
}

/*
Leverages the FeatureStore to extract data and transform it to pandas DataFrame.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
*/
Method DataExtraction(datetime As %String) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    
    try:
        FS = iris.MLpipeline.FeatureStore._New()
        os_rs = FS.DataExtraction("MLpipeline", "PointSamples", "*", f"datetime > '{datetime}'")
        print("Data Extraction ResultSet:", os_rs)
        py_rs = iris.cls("%SYS.Python.SQLResultSet")._New(os_rs)
        df = py_rs.dataframe()
        shape = df.shape
        iris._SYS.System.WriteToConsoleLog(f"DataFrame ready. Shape: {shape}", 0, 0)
        if shape == (0, 0): # Warning for empty result
            iris._SYS.System.WriteToConsoleLog("Query returned an empty DataFrame.", 0, 1)
        return df
    except Exception as e:
        print("Error during Data Extraction:", e)
        iris._SYS.System.WriteToConsoleLog(f"Extraction Error: {str(e)}", 0, 2)
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
        if df is None:
            iris._SYS.System.WriteToConsoleLog("Received None DataFrame for validation.", 0, 2)
            raise ValueError("Input DataFrame is None")
        
        # Missing Values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            iris._SYS.System.WriteToConsoleLog(f"Missing values in DataFrame: {missing_values}", 0, 1)
        
        # Additional validations can be added here (e.g. OUTLIERS)


        # End of validation
        iris._SYS.System.WriteToConsoleLog("DataFrame validation completed successfully.", 0, 0)
        return df
    except Exception as e:
        print("Error during Data Validation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Validation Error: {str(e)}", 0, 2)
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    try:
        if df is None:
            iris._SYS.System.WriteToConsoleLog("Received None DataFrame for preparation.", 0, 2)
            raise ValueError("Input DataFrame is None")
        
        iris._SYS.System.WriteToConsoleLog("Splitting DataFrame into training and test sets.", 0, 0)
        
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(df[["x"]], df["y"], test_size=self.TESTSIZE, random_state=self.SEED)

        iris._SYS.System.WriteToConsoleLog("Data preparation completed successfully.", 0, 0)
        return (Xtrain, Ytrain, Xtest, Ytest)
    except Exception as e:
        print("Error during Data Preparation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Preparation Error: {str(e)}", 0, 2)
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
        if Xtrain is None or Ytrain is None:
            iris._SYS.System.WriteToConsoleLog("Received None Xtrain or Ytrain for model training.", 0, 2)
            raise ValueError("Input data is None")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        mlflow.set_experiment(self._GetParameter("EXPERIMENTNAME"))
        mlflow.sklearn.autolog()

        # Parse Parameters
        if params == "":
            params = self.MODELPARAMS
            iris._SYS.System.WriteToConsoleLog(f"No parameters provided. Using existing ..MODELPARAMS: {params}", 0, 0)
        else:
            self.MODELPARAMS = params
            iris._SYS.System.WriteToConsoleLog(f"Received new parameters for model training: {params}", 0, 0)

        params_dict = json.loads(params._ToJSON())

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression(**params_dict))
        ])

        with mlflow.start_run(run_id=self.MLFLOWRUNID):
            iris._SYS.System.WriteToConsoleLog("Training LinearRegression model with provided training data and parameters.", 0, 0)
            model.fit(Xtrain, Ytrain)
            iris._SYS.System.WriteToConsoleLog("Model training completed successfully.", 0, 0)
            # Calculate metrics
            y_pred_train = model.predict(Xtrain)

            # Log manually with custom keys
            mlflow.log_metrics({
                "train_mae": mean_absolute_error(Ytrain, y_pred_train),
                "train_r2": r2_score(Ytrain, y_pred_train),
            })
            return model

    except Exception as e:

        print("Error during Model Training:", e)

        iris._SYS.System.WriteToConsoleLog(f"Model Training Error: {str(e)}", 0, 2)

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
    set evalResults = AP.ModelEvaluation(model, Xtest, Ytest)
*/
Method ModelEvaluation(model As %SYS.Python, Xtest As %SYS.Python, Ytest As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    import mlflow
    from sklearn.metrics import mean_absolute_error, r2_score
    
    try:
        if model is None:
            iris._SYS.System.WriteToConsoleLog("No trained model available for evaluation.", 0, 2)
            raise ValueError("Model is None")
        if Xtest is None or Ytest is None:
            iris._SYS.System.WriteToConsoleLog("Received None Xtest or Ytest for model evaluation.", 0, 2)
            raise ValueError("Input data is None")

        with mlflow.start_run(run_id=self.MLFLOWRUNID):

            iris._SYS.System.WriteToConsoleLog("Making predictions with the trained model on test data.", 0, 0)
            y_pred_test = model.predict(Xtest)
            
            mae = mean_absolute_error(Ytest, y_pred_test)
            r2 = r2_score(Ytest, y_pred_test)
            mlflow.log_metrics({
                "val_mae": mae,
                "val_r2": r2
            })
            iris._SYS.System.WriteToConsoleLog(f"Model Evaluation completed. MAE: {mae}, R2 Score: {r2}", 0, 0)

            return {"MAE": mae, "R2_Score": r2}

    except Exception as e:
        print("Error during Model Evaluation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Model Evaluation Error: {str(e)}", 0, 2)
        return None
}

/*
Runs the entire ML pipeline sequentially.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    do AP.RunPipeline("2026-02-26 17:50:00")
*/
Method RunPipeline(datetime As %String) [ Language = python ]
{
    import iris
    import pandas as pd
    
    try:
        iris._SYS.System.WriteToConsoleLog("Starting Automated ML Pipeline execution.", 0, 0)
        
        # Step 1: Data Extraction
        iris._SYS.System.WriteToConsoleLog(f"Starting DataExtraction with datetime filter: {datetime}", 0, 0)
        df = self.DataExtraction(datetime)
        
        # Step 2: Data Validation
        iris._SYS.System.WriteToConsoleLog("Starting DataValidation.", 0, 0)
        valid_df = self.DataValidation(df)

        # Step 3: Data Preparation
        iris._SYS.System.WriteToConsoleLog("Starting DataPreparation.", 0, 0)
        preparation_result = self.DataPreparation(valid_df)
        Xtrain, Ytrain, Xtest, Ytest = preparation_result

        # Step 4: Model Training
        iris._SYS.System.WriteToConsoleLog("Starting ModelTraining.", 0, 0)
        model = self.ModelTraining(Xtrain, Ytrain, self.MODELPARAMS)

        # Step 5: Model Evaluation
        iris._SYS.System.WriteToConsoleLog("Starting ModelEvaluation.", 0, 0)
        eval_results = self.ModelEvaluation(model, Xtest, Ytest)

        iris._SYS.System.WriteToConsoleLog(f"Automated ML Pipeline execution completed. Evaluation Results: {eval_results}", 0, 0)
    
    except Exception as e:
        print("Error during Pipeline Execution:", e)
        iris._SYS.System.WriteToConsoleLog(f"Pipeline Execution Error: {str(e)}", 0, 2)
}

}
