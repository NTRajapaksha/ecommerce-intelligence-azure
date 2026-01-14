# Databricks notebook source
# MAGIC %md
# MAGIC Load & Aggregate Data

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

mlflow.set_experiment("/Shared/ecommerce_demand_forecasting")

# 1. Load Data
orders = spark.read.table("ecommerce_silver.orders")
items = spark.read.table("ecommerce_silver.order_items")

# 2. CLEANING: Filter out the "Bad Data" tail (Post August 2018)
# Olist data gets shaky after Aug 2018. We cut it off to preserve model sanity.
orders_clean = orders.filter(col("order_purchase_timestamp") < "2018-08-20")

# 3. Aggregate WEEKLY Sales
# Week is much more stable than Day
weekly_sales = orders_clean.join(items, "order_id") \
    .withColumn("date", to_date(col("order_purchase_timestamp"))) \
    .withColumn("year", year("date")) \
    .withColumn("week_of_year", weekofyear("date")) \
    .groupBy("year", "week_of_year") \
    .agg(
        sum("price").alias("total_sales"), 
        count("order_id").alias("total_orders"),
        min("date").alias("week_start_date") # Keep track of date for plotting
    ) \
    .orderBy("year", "week_of_year")

print("Weekly Sales Aggregated:")
display(weekly_sales)

# COMMAND ----------

# MAGIC %md
# MAGIC Train Forecasting Model (GBT)

# COMMAND ----------

# 1. Feature Engineering
# We only use 'week_of_year' as a seasonal signal
assembler = VectorAssembler(inputCols=["week_of_year", "total_orders"], outputCol="features")

# 2. Split Data
# Train: 2017 to Mid-2018
# Test: Last 10 weeks of reliable data
train_data = weekly_sales.filter(col("week_start_date") < "2018-06-01")
test_data = weekly_sales.filter(col("week_start_date") >= "2018-06-01")

print(f"Train Weeks: {train_data.count()}")
print(f"Test Weeks: {test_data.count()}")

# 3. Train
with mlflow.start_run(run_name="Demand_Forecasting_Weekly"):
    
    # GBT Regressor
    gbt = GBTRegressor(featuresCol="features", labelCol="total_sales", maxIter=50, maxDepth=5)
    
    pipeline = Pipeline(stages=[assembler, gbt])
    
    print("Training Forecasting Model (Weekly)...")
    model = pipeline.fit(train_data)
    
    # 4. Evaluate
    predictions = model.transform(test_data)
    
    evaluator = RegressionEvaluator(labelCol="total_sales", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    
    evaluator_rmse = RegressionEvaluator(labelCol="total_sales", predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    
    mlflow.log_metric("r2", r2)
    mlflow.spark.log_model(model, "forecasting-model")
    
    print(f"✅ Model Trained.")
    print(f"R2 Score: {r2:.4f} (Target: > 0.0)")
    print(f"RMSE: ${rmse:.2f}")

    # 5. Save Forecasts to Gold
    predictions.withColumn("model_version", lit("GBT_Weekly_v2")) \
        .write.format("delta").mode("overwrite").saveAsTable("ecommerce_gold.demand_forecasts")
    print("Forecasts saved to Gold Layer.")

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize & Save Results

# COMMAND ----------

# 1. Visualize Actual vs Predicted
# This is crucial for your Dashboard
display(predictions.select("week_start_date", "total_sales", "prediction").orderBy("week_start_date"))

# 2. Save Forecasts to Gold for Power BI
predictions.withColumn("model_version", lit("GBT_v1")) \
    .write.format("delta").mode("overwrite").saveAsTable("ecommerce_gold.demand_forecasts")

print("Forecasts saved to Gold Layer.")

# COMMAND ----------

