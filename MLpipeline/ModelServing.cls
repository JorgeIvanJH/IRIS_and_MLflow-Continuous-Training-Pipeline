Class MLpipeline.ModelServing Extends %RegisteredObject
{

/*
Loads the model and serves it for inference. 
Caches the model in memory and reloads it only when the path changes, as determined by a global variable.
e.g:
    do ##class(MLpipeline.ModelServing).Predict(X)
*/
ClassMethod Predict(inputData As %String) As %SYS.Python [ Language = python ]
{
    import iris
    import mlflow
    import pandas as pd
    import json
    global _cached_model, _current_path

    try:
        gref = iris.gref("^ML.Config")
        new_model_path = gref["ModelPath"]

        if not new_model_path:
            iris._SYS.System.WriteToConsoleLog("No model path found in ^ML.Config('ModelPath')", 0, 2)
            raise Exception("No model path found in ^ML.Config('ModelPath')")

        # Load the model if it's not cached or if the path has changed
        if '_cached_model' not in globals() or _current_path != new_model_path:
            _cached_model = mlflow.sklearn.load_model(new_model_path)
            _current_path = new_model_path
            iris._SYS.System.WriteToConsoleLog(f"New model loaded from {new_model_path}", 0, 0)

        iris._SYS.System.WriteToConsoleLog(f"Predicting with model from {new_model_path}", 0, 0)
        data_dict = json.loads(inputData)
        X = pd.DataFrame(data_dict['data'], columns=data_dict['columns'])
        preds = _cached_model.predict(X)
        y_pred = pd.DataFrame(preds, columns=['y_pred'], index=X.index)
        return y_pred.to_json(orient='split')

    except Exception as e:
        err_msg = f"Predict Error: {str(e)}"
        print(f"Predict Error: {err_msg}")
        iris._SYS.System.WriteToConsoleLog(f"Predict Error: {err_msg}", 0, 2)
        return json.dumps({"error": err_msg})
}

}
