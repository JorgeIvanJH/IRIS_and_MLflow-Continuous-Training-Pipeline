Class MLpipeline.PredictionService Extends %RegisteredObject
{

/*
Extracts data from IRIS for prediction based on the input filters. The filters should be a valid SQL WHERE clause without the 'WHERE' keyword.
e.g:
    set df = ##class(MLpipeline.PredictionService).PredDataExtraction("datetime > '2026-05-14 00:00:00'")
*/
ClassMethod PredDataExtraction(filters As %String = "") As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    from python_utils.utils import IRIS_DBQuery

    try:
        df = IRIS_DBQuery(
            schema="MLpipeline", 
            tablename="PointSamples", 
            columns="ID, x, y, label, datetime",
            filters=filters
        )
        return df
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"PredDataExtraction Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"PredDataExtraction Error: {msg}", 0, 2)
        return pd.DataFrame()
}

/*
Loads the model and serves it for inference. 
Caches the model in memory and reloads it only when the path changes, as determined by a global variable.
e.g:
    set model = ##class(MLpipeline.PredictionService).CachedModelLoading()
*/
ClassMethod CachedModelLoading() As %SYS.Python [ Language = python ]
{
    import iris
    from python_utils.utils import safe_model_load

    if not hasattr(iris.cls(__name__), '_cached_model'):
        iris.cls(__name__)._cached_model = None
        iris.cls(__name__)._current_path = None

    try:
        gref = iris.gref("^ML.Config")
        model_path = gref["ModelPath"]

        # Only reload if path changed or model is missing
        if iris.cls(__name__)._current_path != model_path or iris.cls(__name__)._cached_model is None:
            iris.cls(__name__)._cached_model = safe_model_load(model_path)
            iris.cls(__name__)._current_path = model_path
            iris._SYS.System.WriteToConsoleLog(f"Model reloaded from {model_path}", 0, 0)

        return iris.cls(__name__)._cached_model
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"CachedModelLoading Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"CachedModelLoading Error: {msg}", 0, 2)
        return None
}

/*
Uploads a CSV file to an IRIS table. It performs basic validation and type inference, including date parsing.
*/
ClassMethod SavePredictions(model As %SYS.Python, indf As %SYS.Python, labelpreds As %SYS.Python, predtime As %Double, namespace As %String = "USER") As %Status [ Language = python ]
{
    from sqlalchemy import create_engine
    import pandas as pd
    import iris
    
    try:
        engine = create_engine(f"iris+emb:///{namespace}")

        numrows, numcols = indf.shape

        df_save = pd.DataFrame({})
        # df_save["ID"] will be auto-generated in IRIS as an identity column
        df_save["PointsIDForeignKey"] = indf["id"]
        df_save["LabelPred"] = labelpreds
        df_save["Modelrunid"] = model.mlflow_model_info.metadata.get("run_id")
        df_save["InferenceTime"] = predtime/numrows if numrows > 0 else 0
        # df_save["datetime"] populated by default in IRIS with CURRENT_TIMESTAMP

        # Append data to the existing table
        df_save.to_sql("Predictions", engine, schema="MLpipeline", if_exists='append', index=False, method='multi')
        iris._SYS.System.WriteToConsoleLog(f"Successfully saved {numrows} predictions to MLpipeline.Predictions", 0, 0)
        return True

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"SavePredictions Error: {msg}")
        iris._SYS.System.WriteToConsoleLog(f"SavePredictions Error: {msg}", 0, 2)
        return False
}

/*
Predicts over the data specified by the filters using the current model in production and saves the predictions back to IRIS. The filters should be a valid SQL WHERE clause without the 'WHERE' keyword.
e.g:
    set predictionsjson = ##class(MLpipeline.PredictionService).Predict("datetime > '2026-05-14 00:00:00'")
*/
ClassMethod Predict(filters As %String = "") As %String [ Language = python ]
{
    import iris
    import json
    from python_utils.utils import measure_time_decorator


    try:
        iris._SYS.System.WriteToConsoleLog(f"Starting prediction with filters: {filters}", 0, 0)
        indf = iris.cls(__name__).PredDataExtraction(filters)
        model = iris.cls(__name__).CachedModelLoading()
        decorated_predict = measure_time_decorator(model.predict)
        X = indf[["x", "y"]]
        X["year"] = indf["datetime"].dt.year
        X["month"] = indf["datetime"].dt.month
        X["day"] = indf["datetime"].dt.day
        X["hour"] = indf["datetime"].dt.hour
        X["minute"] = indf["datetime"].dt.minute
        X["second"] = indf["datetime"].dt.second
        label_preds, pred_time = decorated_predict(X)
        iris.MLpipeline.PredictionService.SavePredictions(model, indf, label_preds, pred_time)
        iris._SYS.System.WriteToConsoleLog(f"Prediction completed in {pred_time:.4f} seconds for {len(indf)} records.", 0, 0)
        return json.dumps(label_preds.tolist() if hasattr(label_preds, 'tolist') else list(label_preds))
    except Exception as e:
        err_msg = f"Predict Error: {str(e)}"
        print(f"Predict Error: {err_msg}")
        iris._SYS.System.WriteToConsoleLog(f"Predict Error: {err_msg}", 0, 2)
        return json.dumps([None])
}

}
