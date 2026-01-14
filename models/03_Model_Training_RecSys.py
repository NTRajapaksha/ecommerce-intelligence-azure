# Databricks notebook source
# MAGIC %md
# MAGIC Imports & Configuration

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Set Experiment Name (Organizes runs in the Experiments tab)
mlflow.set_experiment("/Shared/ecommerce_recommendations")

print("Libraries loaded & MLflow experiment set.")

# COMMAND ----------

# MAGIC %md
# MAGIC Load Data & Preprocessing (String Indexing)

# COMMAND ----------

# 1. Load the Gold Interaction Matrix
df_interactions = spark.read.table("ecommerce_gold.interaction_matrix")

# 2. CLEANING: Remove rows where customer or product is NULL
# (ALS cannot learn from interaction with "nothing")
df_clean = df_interactions.filter(
    (col("customer_unique_id").isNotNull()) & 
    (col("product_id").isNotNull()) &
    (col("product_id") != "nan")
)

# 3. Convert UUIDs to Integers (Required for ALS)
# Added handleInvalid="skip" to safely ignore any new weird edge cases during training
indexer_user = StringIndexer(
    inputCol="customer_unique_id", 
    outputCol="user_int", 
    handleInvalid="skip"
)

indexer_item = StringIndexer(
    inputCol="product_id", 
    outputCol="item_int", 
    handleInvalid="skip"
)

pipeline_prep = Pipeline(stages=[indexer_user, indexer_item])

# Fit & Transform on the CLEAN data
model_prep = pipeline_prep.fit(df_clean)
df_final = model_prep.transform(df_clean)

# 4. Split Data
(train_data, test_data) = df_final.randomSplit([0.8, 0.2], seed=42)

print(f"Original Count: {df_interactions.count()}")
print(f"Cleaned Count: {df_clean.count()}")
print(f"Training Count: {train_data.count()}")
print(f"Test Count: {test_data.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC Train Model with MLOps (MLflow)

# COMMAND ----------

# Start an MLflow Run to track this training session
with mlflow.start_run(run_name="ALS_Collaborative_Filtering"):
    
    # 1. Configure Model
    # Rank: Number of latent factors (hidden features)
    # MaxIter: Number of passes over data
    # RegParam: Regularization to prevent overfitting
    als = ALS(
        maxIter=10, 
        regParam=0.1, 
        rank=10,
        userCol="user_int", 
        itemCol="item_int", 
        ratingCol="interaction_strength",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=True 
    )
    
    # 2. Train
    print("Training ALS Model...")
    model = als.fit(train_data)
    
    # 3. Evaluate
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(
        metricName="rmse", 
        labelCol="interaction_strength",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    
    # 4. Log Metrics & Parameters to MLflow
    mlflow.log_param("maxIter", 10)
    mlflow.log_param("regParam", 0.1)
    mlflow.log_param("rank", 10)
    mlflow.log_metric("rmse", rmse)
    
    # 5. Log the Model itself (Save it to the Model Registry)
    mlflow.spark.log_model(model, "als-model")
    
    print(f"✅ Training Complete. RMSE: {rmse}")
    print("Model saved to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC Generate Recommendations for All Users

# COMMAND ----------

# 1. Generate top 10 product recommendations for each user
print("Generating recommendations...")
user_recs = model.recommendForAllUsers(10)

# The output 'recommendations' is an array of structs. We need to explode it.
from pyspark.sql.functions import explode

df_recs_exploded = user_recs \
    .withColumn("rec", explode("recommendations")) \
    .select(
        col("user_int"), 
        col("rec.item_int").alias("item_int"), 
        col("rec.rating").alias("prediction")
    )

# 2. Convert Integer IDs back to real UUID Strings
# We use the metadata from the earlier StringIndexers to reverse map
# (This is a bit complex in Spark, but essential for the final table)

# Get labels from metadata
user_labels = model_prep.stages[0].labels
item_labels = model_prep.stages[1].labels

from pyspark.sql.functions import pandas_udf
import pandas as pd

# Broadcast labels for lookup efficiency (if dataset was huge, we'd use a join, but this fits in memory)
# Simpler approach: Join back to the mapping tables
user_mapping = df_final.select("user_int", "customer_unique_id").distinct()
item_mapping = df_final.select("item_int", "product_id").distinct()

final_recs = df_recs_exploded \
    .join(user_mapping, "user_int") \
    .join(item_mapping, "item_int") \
    .select("customer_unique_id", "product_id", "prediction")

# 3. Save to Gold
final_recs.write.format("delta").mode("overwrite").saveAsTable("ecommerce_gold.product_recommendations")

print("✅ Recommendations saved to ecommerce_gold.product_recommendations")
display(final_recs.limit(10))

# COMMAND ----------

