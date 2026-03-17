Class MLpipeline.ModelServing Extends %RegisteredObject
{

// TODO: MEthod to download model from mlflow if doesnt exist in path

/*
Updates the global containing the training model's path to be used by the PredictionService component for inference. 
e.g:
    set status = ##class(MLpipeline.ModelServing).DeployModel("/path/to/new/model")
*/
ClassMethod DeployModel(newPath As %String) As %Status
{
    set ^ML.Config("ModelPath") = newPath
    do ##class(%SYS.System).WriteToConsoleLog("Model path updated to: "_newPath, 0, 0)
    quit $$$OK
}

/*
Contains the logic for promoting a new model to production based on the validation results in MLpipeline.ModelValidation. 
newrunid: is the most recent model run ID
bestrunid: is the best performing model run ID based on validation metrics. It can be the newrunid, or the current model's run ID in production.
^ML.Config("ModelPath"): contains the path to the weights of the current production model.
The new run model is promoted if it outperforms the current production model, and if user approves.
e.g:
    set status = ##class(MLpipeline.ModelServing).PromotingLogic("b4c22ea287704278a351f934ab0038b5", "b4c22ea287704278a351f934ab0038b5")
*/
ClassMethod PromotingLogic(newrunid As %String, bestrunid As %String) As %SYS.Python [ Language = python ]
{
    import os
    import iris
    import mlflow
    try:
        gref = iris.gref("^ML.Config")
        active_path = gref["ModelPath"]
        
        candidate_model_path = os.path.join(iris.MLpipeline.AutomatedPipeline._GetParameter("MODELSPATH"), bestrunid)

        if active_path:
            # A model already exists. Compare performance.
            if bestrunid == newrunid:
                input_promote = input("New model outperforms the baseline. Promote? (yes/no): ")
                if input_promote.lower() == "yes":
                    iris._SYS.System.WriteToConsoleLog("Promotion approved. Deploying new model.", 0, 0)
                    iris.MLpipeline.ModelServing.DeployModel(candidate_model_path)
                else:
                    iris._SYS.System.WriteToConsoleLog("Promotion declined. Keeping old model.", 0, 0)
            else:
                iris._SYS.System.WriteToConsoleLog("New model failed to beat baseline. Keeping old model.", 0, 0)
        else:
            # Cold start. No model in ^ML.Config.
            iris._SYS.System.WriteToConsoleLog("No existing model found. Deploying first model.", 0, 0)
            iris.MLpipeline.ModelServing.DeployModel(candidate_model_path)
        return True

    except Exception as e:
        print(f"PromotingLogic Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"PromotingLogic Error: {str(e)}", 0, 2)
        return False
}

}
