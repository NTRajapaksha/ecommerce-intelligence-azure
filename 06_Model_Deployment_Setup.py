# Databricks notebook source
import mlflow
import mlflow.spark

# 1. Load the trained ALS Model
# We load it from the run we just did (or you can grab the latest active run)
# Note: In production, we'd use the Run ID, but for this demo, we'll grab the latest logged model
run_id = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name("/Shared/ecommerce_recommendations").experiment_id]) \
    .sort_values("start_time", ascending=False).iloc[0]["run_id"]

model_uri = f"runs:/{run_id}/als-model"
print(f"Found Best Model URI: {model_uri}")

# 2. Register the Model in the Azure ML Model Registry
# This makes it visible in the "Models" tab of Azure Machine Learning Studio
model_name = "olist-recsys-v1"

print(f"Registering model '{model_name}'...")
mv = mlflow.register_model(model_uri, model_name)

print(f"✅ Model registered! Version: {mv.version}")
print("Go to Azure Portal -> Machine Learning Workspace -> Models to see it.")

# COMMAND ----------

