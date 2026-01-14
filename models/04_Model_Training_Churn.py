# Databricks notebook source
# MAGIC %md
# MAGIC Imports & Config

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import mlflow
import mlflow.spark

# Set Experiment
mlflow.set_experiment("/Shared/ecommerce_churn_prediction")

print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC Load Data & Feature Engineering

# COMMAND ----------

# 1. Load Customer 360 Features
df_churn = spark.read.table("ecommerce_gold.customer_360")

# 2. Select Features for Training
# CRITICAL: We DROP 'recency_days' to prevent Data Leakage.
# We want to predict risk based on spending habits, not just "time since last seen".
feature_cols = [
    "frequency", 
    "total_monetary", 
    "avg_basket_value", 
    "total_items_bought"
]

print(f"Training with features: {feature_cols}")

# 3. Assemble Features into a Vector (Required for Spark ML)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")

# 4. Scale Features (Good practice for spending data which varies wildly)
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

# 5. Split Data
# Stratified split helps if churn labels are unbalanced, but random is okay for this size
(train_data, test_data) = df_churn.randomSplit([0.8, 0.2], seed=42)

print(f"Train Size: {train_data.count()}")
print(f"Test Size: {test_data.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC Train Gradient Boosted Tree (GBT) Model

# COMMAND ----------

with mlflow.start_run(run_name="Churn_Prediction_GBT"):
    
    # 1. Define Model
    # GBT is similar to XGBoost (Gradient Boosting)
    gbt = GBTClassifier(
        labelCol="churn_risk_label", 
        featuresCol="features",
        maxIter=20,
        maxDepth=5
    )
    
    # 2. Build Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, gbt])
    
    # 3. Train
    print("Training Churn Model...")
    model = pipeline.fit(train_data)
    
    # 4. Evaluate
    predictions = model.transform(test_data)
    
    # Area Under ROC (AUC) is better for classification than Accuracy
    evaluator = BinaryClassificationEvaluator(
        labelCol="churn_risk_label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    
    # 5. Log to MLflow
    mlflow.log_metric("auc", auc)
    mlflow.spark.log_model(model, "churn-model")
    
    print(f"✅ Model Trained. AUC Score: {auc:.4f}")
    
    # Feature Importance (Explainability)
    # Extract the GBT model from the pipeline (it's the 3rd stage, index 2)
    gbt_model = model.stages[2]
    importances = gbt_model.featureImportances
    
    print("\nFeature Importances:")
    for i, col_name in enumerate(feature_cols):
        print(f"{col_name}: {importances[i]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Run Inference (Identify At-Risk Customers)

# COMMAND ----------

# Apply model to ALL customers
full_predictions = model.transform(df_churn)

# Filter for High Risk customers who are valuable (High Monetary Value)
# "High Risk" = Prediction 1
high_risk_customers = full_predictions \
    .filter(col("prediction") == 1) \
    .filter(col("total_monetary") > 100) \
    .select("customer_id", "total_monetary", "recency_days", "probability") \
    .orderBy(col("total_monetary").desc())

print("⚠️ Top 10 High-Value Customers at Risk of Churning:")
display(high_risk_customers.limit(10))

# Save predictions to Gold for the Dashboard
high_risk_customers.write.format("delta").mode("overwrite").saveAsTable("ecommerce_gold.churn_predictions")
