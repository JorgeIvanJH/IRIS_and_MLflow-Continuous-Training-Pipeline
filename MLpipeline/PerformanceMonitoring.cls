Class MLpipeline.PerformanceMonitoring Extends %RegisteredObject
{

// Experiment Name for tracking in MLflow

Parameter EXPERIMENTNAME = "PerformanceMonitoring";

// Minimum F1 score threshold for model performance to trigger retraining

Parameter F1THRESHOLD = 0.7;

// Current Date and Time for logging purposes

Property CURRENTDATETIME As %String;

// Run id for MLflow tracking

Property MLFLOWRUNID As %String;

/*
Set default values of parameters when a new instance of AutomatedPipeline is created.
e.g:
    set PM = ##class(MLpipeline.PerformanceMonitoring).%New()
*/
Method %OnNew() As %Status [ Private ]
{
    Set ..CURRENTDATETIME = $ZDATETIME($ZTIMESTAMP, 3, 1, 3)
    try {
        set mlflow = ##class(%SYS.Python).Import("mlflow")
        do mlflow."set_tracking_uri"($System.Util.GetEnviron("MLFLOW_TRACKING_URI_IRIS"))
        do mlflow."set_experiment"(..#EXPERIMENTNAME)
        do mlflow."end_run"() // Ensure no active run before starting a new one
        set run = mlflow."start_run"()
        set runid = run.info."run_id"
        do mlflow."end_run"()
        set ..MLFLOWRUNID = runid
        do ##class(%SYS.System).WriteToConsoleLog("Created MLflow run with id="_runid, 0, 0)
    } catch ex {
        set status = ex.AsStatus()
        do ##class(%SYS.System).WriteToConsoleLog("MLflow Error: "_ex.DisplayString(), 0, 2)
    }

    Return $$$OK
}

/*
Gets Ground Truth and Predictions from IRIS tables, and processes for performance monitoring.
Analysis is done only on points where ground truth is available.
e.g:
    set PM = ##class(MLpipeline.PerformanceMonitoring).%New()
    set tables = PM.FetchTables("2026-05-14 00:00:00")
*/
Method FetchTables(fromdatetime As %String) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    import datetime as dt
    from python_utils.utils import IRIS_DBQuery

    try:
        parsed_dt = dt.datetime.strptime(fromdatetime, "%Y-%m-%d %H:%M:%S")
        safe_literal = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
        PointSamples_df = IRIS_DBQuery(
            schema="MLpipeline", 
            tablename="PointSamples", 
            columns="id, x, y, label, datetime",
            filters=f"datetime >= '{safe_literal}' AND datetime <= '{self.CURRENTDATETIME}'"
        )
        points_w_groundtruth = PointSamples_df.loc[PointSamples_df["y"].notna(), "id"]
        Predictions_df = IRIS_DBQuery(
            schema="MLpipeline", 
            tablename="Predictions", 
            columns="id, pointsidforeignkey, ypred, modelrunid, inferencetime, datetime",
            filters=f"pointsidforeignkey IN {tuple(points_w_groundtruth)}"
        )
        return PointSamples_df, Predictions_df
    except Exception as e:
        print(f"FetchTables Error: {str(e)}")
        #; iris._SYS.System.WriteToConsoleLog(f"FetchTables Error: {str(e)}", 0, 2)
        return pd.DataFrame(), pd.DataFrame()
}

/*
Triggers the retraining pipeline if performance degradation is detected based on the F1 score.
e.g:
    set PM = ##class(MLpipeline.PerformanceMonitoring).%New()
    set status = PM.Trigger(0.15, "2026-05-14 00:00:00")
*/
Method Trigger(f1 As %Numeric, fromdatetime As %String) As %Status [ Language = python ]
{
    import iris
    minf1threshold = float(iris.cls(__name__)._GetParameter("F1THRESHOLD"))

    if f1 < minf1threshold:
        iris._SYS.System.WriteToConsoleLog(f"[PERFORMANCE MONITORING] Performance degradation detected! F1 score {f1} is below the threshold of {minf1threshold}. RETRAINING...", 0, 0)
        AP = iris.MLpipeline.AutomatedPipeline._New()
        summary = AP.RunPipeline(fromdatetime)
        iris._SYS.System.WriteToConsoleLog(f"[PERFORMANCE MONITORING] Retraining pipeline triggered. Summary: {summary}", 0, 0)
    else:
        iris._SYS.System.WriteToConsoleLog(f"[PERFORMANCE MONITORING] Model performance is stable. F1 score: {f1} is above the threshold of {minf1threshold}.", 0, 0)
}

/*
Makes model evaluation and logs metrics and custom plots to MLflow.
e.g:
    set PM = ##class(MLpipeline.PerformanceMonitoring).%New()
    set status = PM.MetricsMonitoring("2026-05-14 00:00:00")
*/
Method MetricsMonitoring(fromdatetime As %String) As %Status [ Language = python ]
{

    import iris
    import mlflow
    import pandas as pd
    from sklearn.metrics import f1_score
    from python_utils.utils import IRIS_DBQuery
    import matplotlib.pyplot as plt
    import seaborn as sns



    def custom_plot(PointSamples_df, Predictions_df):
        eachmodelspred = Predictions_df.pivot_table(index="pointsidforeignkey", columns="modelrunid", values="ypred", aggfunc="mean")
        merged_df = pd.merge(PointSamples_df, eachmodelspred, left_on="id", right_on="pointsidforeignkey", how="left")
        modelnames = [col for col in merged_df.columns if col not in PointSamples_df.columns]
        fig, ax = plt.subplots(figsize=(12, 6))
        for colname in modelnames + ["y"]:
            sns.scatterplot(data=merged_df, x="x", y=colname, label=str(colname), ax=ax)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_title("Predictions Over Time by Model Run")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Y")
        ax.legend(title="Model Run id")
        ax.grid(True)
        fig.tight_layout()

        return fig

    try:
        PointSamples_df, Predictions_df = self.FetchTables(fromdatetime)
        if PointSamples_df.empty or Predictions_df.empty:
            print("No data available for metrics monitoring.")
            return False
        models_last_datetime = Predictions_df.groupby("modelrunid").agg({"datetime":max})
        latest_model = models_last_datetime["datetime"].idxmax()
        latest_active_predictions = Predictions_df[Predictions_df["modelrunid"] == latest_model].groupby("pointsidforeignkey").agg({"ypred": "mean", "inferencetime": "mean"}).reset_index()
        latest_active_predictions = pd.merge(latest_active_predictions, PointSamples_df[["id", "y"]], left_on="pointsidforeignkey", right_on="id", how="left")

        with mlflow.start_run(run_id=self.MLFLOWRUNID):

            f1 = f1_score(latest_active_predictions["y"], latest_active_predictions["ypred"], average="weighted")

            mlflow.log_metric("F1", f1)

            modelrun = mlflow.get_run(latest_model)
            mlflow.set_tag("model_id", latest_model)
            mlflow.set_tag("run_name", modelrun.data.tags.get('mlflow.runName'))

            fig = custom_plot(PointSamples_df, Predictions_df)
            mlflow.log_figure(fig, "plots/performance_monitoring.png")
            plt.close(fig)

        self.Trigger(f1, fromdatetime) # Trigger retraining if performance degradation is detected

        return True

    except Exception as e:
        print(f"MetricsMonitoring Error: {str(e)}")
        #; iris._SYS.System.WriteToConsoleLog(f"MetricsMonitoring Error: {str(e)}", 0, 2)
        return False
}

}
