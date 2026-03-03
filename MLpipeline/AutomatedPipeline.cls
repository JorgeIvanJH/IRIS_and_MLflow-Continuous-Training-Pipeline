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
Prepares the validated DataFrame for model training
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
        
        if not isinstance(self.SCALER, StandardScaler):
            iris._SYS.System.WriteToConsoleLog("No existing StandardScaler found. Initializing new scaler and fitting to training data.", 0, 0)
            self.SCALER = StandardScaler()
            self.SCALER.fit(Xtrain)
        else:
            iris._SYS.System.WriteToConsoleLog("Using existing StandardScaler for normalization.", 0, 0)
        
        iris._SYS.System.WriteToConsoleLog("Applying StandardScaler transformation to training and test data.", 0, 0)
        Xtrain = self.SCALER.transform(Xtrain)
        Xtest = self.SCALER.transform(Xtest)

        iris._SYS.System.WriteToConsoleLog("Data preparation completed successfully.", 0, 0)
        return (Xtrain, Ytrain, Xtest, Ytest)
    except Exception as e:
        print("Error during Data Preparation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Preparation Error: {str(e)}", 0, 2)
        return None
}

/*
Trains a machine learning model (e.g., LinearRegression) using the prepared data and specified parameters.
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
        self.MODEL = LinearRegression(**params_dict)
        
        iris._SYS.System.WriteToConsoleLog("Training LinearRegression model with provided training data and parameters.", 0, 0)
        self.MODEL.fit(Xtrain, Ytrain)
        iris._SYS.System.WriteToConsoleLog("Model training completed successfully.", 0, 0)

        return self.MODEL

    except Exception as e:

        print("Error during Model Training:", e)

        iris._SYS.System.WriteToConsoleLog(f"Model Training Error: {str(e)}", 0, 2)

        return None
}

}
