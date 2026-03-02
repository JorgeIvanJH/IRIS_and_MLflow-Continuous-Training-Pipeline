Class MLpipeline.AutomatedPipeline Extends %RegisteredObject
{

Method %OnNew() As %Status [ Private ]
{
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.AutomatedPipeline", 0, 0)
    Quit $$$OK
}

/// Leverages the FeatureStore to extract data and transform it to pandas DataFrame.
/// e.g:
///     set AP = ##class(MLpipeline.AutomatedPipeline).%New()
///     set df = AP.DataExtraction("2026-02-26 17:50:00")
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

/// Validates the extracted DataFrame for missing values and logs the results.
/// e.g:
///     set AP = ##class(MLpipeline.AutomatedPipeline).%New()
///     set df = AP.DataExtraction("2026-02-26 17:50:00")
///     set validDf = AP.DataValidation(df)
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
        
        # Additional validations can be added here


        # End of validation
        iris._SYS.System.WriteToConsoleLog("DataFrame validation completed successfully.", 0, 0)
        return df
    except Exception as e:
        print("Error during Data Validation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Validation Error: {str(e)}", 0, 2)
        return None
}

/// Prepares the validated DataFrame for model training
/// e.g:
///     set AP = ##class(MLpipeline.AutomatedPipeline).%New()
///     set df = AP.DataExtraction("2026-02-26 17:50:00")
///     set validDf = AP.DataValidation(df)
///     set prepDf = AP.DataPreparation(validDf)
Method DataPreparation(df As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    try:
        if df is None:
            iris._SYS.System.WriteToConsoleLog("Received None DataFrame for preparation.", 0, 2)
            raise ValueError("Input DataFrame is None")
        
        # Normalization
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        iris._SYS.System.WriteToConsoleLog(f"StandardScaler Normalized columns: {list(numeric_cols)}", 0, 0)

        # Additional preparation steps can be added here
        

        # End of preparation
        iris._SYS.System.WriteToConsoleLog("DataFrame preparation completed successfully.", 0, 0)
        return df
    except Exception as e:
        print("Error during Data Preparation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Preparation Error: {str(e)}", 0, 2)
        return None
}

}
