import os
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import pandas as pd
import lightgbm as lgb
import time


def measure_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    return wrapper


def plot_inference(self, Xtrain, Ytrain, Xtest, Ytest, oldrun, newrun):
    """
    Plots the inference results of the old and new models along with the training and testing data.
    Assumes the path to the models' file is defined in the MODELSPATH parameter in an objectscript class
    Args:
        self: The instance of the class calling this function, used to access parameters.
        Xtrain (pd.DataFrame): Training features.
        Ytrain (pd.Series): Training labels.
        Xtest (pd.DataFrame): Testing features.
        Ytest (pd.Series): Testing labels.
        oldrun (mlflow.entities.Run): The MLflow run object for the old model.
        newrun (mlflow.entities.Run): The MLflow run object for the new model.
    """
    oldrunname = oldrun.get("tags.mlflow.runName")
    newrunname = newrun.get("tags.mlflow.runName")

    oldmodel = mlflow.sklearn.load_model(
        os.path.join(eval("""self._GetParameter("MODELSPATH")"""), oldrun.run_id)
    )
    newmodel = mlflow.sklearn.load_model(
        os.path.join(eval("""self._GetParameter("MODELSPATH")"""), newrun.run_id)
    )

    line_x = np.linspace(Xtest.min(), Xtest.max(), 100).reshape(-1, 1)
    line_y_old = oldmodel.predict(line_x)
    line_y_new = newmodel.predict(line_x)
    plt.figure(figsize=(10, 6))
    if not Xtrain.empty and not Ytrain.empty:
        plt.scatter(Xtrain, Ytrain, color="orange", label="Train Data")
    if not Xtest.empty and not Ytest.empty:
        plt.scatter(Xtest, Ytest, color="blue", label="Test Data")
    plt.plot(line_x, line_y_old, color="red", label=f"Old Model: {oldrunname}")
    plt.plot(line_x, line_y_new, color="green", label=f"New Model: {newrunname}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid()
    plt.savefig(f"/dur/log/model_comparison_{oldrunname}_vs_{newrunname}.png")
    plt.close()

def save_mlflow_model(runid: str):
    """
    Loads a model from MLflow using the provided run ID and re-saves it to the path specified in the MODELSPATH parameter.
    """
    import iris
    import mlflow
    import os
    import dotenv
    dotenv.load_dotenv()

    try:
        iris._SYS.System.WriteToConsoleLog(f"Attempting to re-save model for Run ID: {runid}", 0, 0)
        # Use the internal Docker network URL for the MLflow container
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI_IRIS"))
        model_uri = f"runs:/{runid}/model"
        model = mlflow.sklearn.load_model(model_uri)
        base_path = iris.cls("MLpipeline.AutomatedPipeline")._GetParameter("MODELSPATH")
        model_path = os.path.join(base_path, runid)
        mlflow.sklearn.save_model(model, path=model_path)
        iris._SYS.System.WriteToConsoleLog(f"Model re-saved to: {model_path}", 0, 0)
        return True
    except Exception as e:
        print(f"ReSaveMLflowModel Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"ReSaveMLflowModel Error: {str(e)}", 0, 2)
        return False

def safe_model_load(model_path: str):
    """
    Safely loads a model from the specified path. If loading fails, it attempts to re-save the model and load it again.
    """
    import iris
    import mlflow
    import os

    try:
        model = mlflow.sklearn.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"Error loading model from {model_path}: {str(e)}", 0, 2)
        # Extract run ID from the model path and attempt to re-save the model
        run_id = os.path.basename(model_path)
        print(f"Attempting to re-save model for Run ID: {run_id}")
        iris._SYS.System.WriteToConsoleLog(f"Attempting to re-save model for Run ID: {run_id}", 0, 0)
        if save_mlflow_model(run_id):
            try:
                print(f"Attempting to load model again from {model_path} after re-saving.")
                iris._SYS.System.WriteToConsoleLog(f"Attempting to load model again from {model_path} after re-saving.", 0, 0)
                model = mlflow.sklearn.load_model(model_path)
                return model
            except Exception as e:
                print(f"Error loading model after re-saving from {model_path}: {str(e)}")
                iris._SYS.System.WriteToConsoleLog(f"Error loading model after re-saving from {model_path}: {str(e)}", 0, 2)
                return None
        else:
            return None