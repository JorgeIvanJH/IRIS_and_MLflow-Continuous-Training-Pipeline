import mlflow

# Set tracking server 
mlflow.set_tracking_uri("http://localhost:5000") # Can also be set using environment variable MLFLOW_TRACKING_URI
mlflow.set_experiment("my-first-experiment") # Can also be set using environment variable MLFLOW_EXPERIMENT_NAME



# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")

# Test logging
with mlflow.start_run():
    mlflow.log_param("test_param", "test_value")
    print("✓ Successfully connected to MLflow!")