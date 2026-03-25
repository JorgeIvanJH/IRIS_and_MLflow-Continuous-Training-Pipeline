import os
import dotenv
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.lightgbm

dotenv.load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


RUN_ID = "run id from the experiment" # e.g: 23120aba7b614f2eaa3e0ec15b10c6e9


X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # SAME seed
)

model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.lightgbm.load_model(model_uri)

y_pred = model.predict(X_test)
rmse_new = np.sqrt(mean_squared_error(y_test, y_pred))

print("Recomputed RMSE:", rmse_new)


# # Or when downloading the model manually from the MLflow UI, you can load it like this:
# import joblib
# model_path = "model.pkl"
# model = joblib.load(model_path)
# y_pred = model.predict(X_test)
# rmse_new = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Recomputed RMSE from local model:", rmse_new)
