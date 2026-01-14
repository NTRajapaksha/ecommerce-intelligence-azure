# Databricks notebook source
# MAGIC %md
# MAGIC Setup & Load Silver Data

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Define paths (Adjust if your storage account name is different)
storage_name = "AZURE_STORAGE_ACCOUNT_NAME" # REPLACE with your actual storage account name
silver_path = f"abfss://silver@{storage_name}.dfs.core.windows.net"

# Load Silver Tables
orders = spark.read.table("ecommerce_silver.orders")
items = spark.read.table("ecommerce_silver.order_items")
customers = spark.read.table("ecommerce_silver.customers")
clickstream = spark.read.table("ecommerce_silver.clickstream")

print("Silver tables loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Set 1 - Customer 360 (For Churn & CLV Models)

# COMMAND ----------

# 1. Join Orders and Items to get transaction details
transactions = orders.join(items, "order_id", "inner")

# 2. Calculate RFM (Recency, Frequency, Monetary)
max_date = orders.select(max("order_purchase_timestamp")).collect()[0][0]

customer_features = transactions.groupBy("customer_id") \
    .agg(
        countDistinct("order_id").alias("frequency"),
        sum("price").alias("total_monetary"),
        avg("price").alias("avg_basket_value"),
        max("order_purchase_timestamp").alias("last_purchase_date"),
        count("product_id").alias("total_items_bought")
    ) \
    .withColumn("recency_days", datediff(lit(max_date), col("last_purchase_date")))

# 3. Add Churn Label (If inactive > 90 days = 1)
customer_features = customer_features \
    .withColumn("churn_risk_label", when(col("recency_days") > 90, 1).otherwise(0))

# 4. Save to Gold
spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_gold")
customer_features.write.format("delta").mode("overwrite").saveAsTable("ecommerce_gold.customer_360")

print("✅ Customer 360 features created.")
display(customer_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Set 2 - User-Item Interaction Matrix (For Recommendations)

# COMMAND ----------

# Define Weights
# View = 1, Add to Cart = 3, Purchase = 5
events = clickstream.select("customer_unique_id", "product_id", "event_type") \
    .withColumn("score", 
        when(col("event_type") == "product_view", 1)
        .when(col("event_type") == "add_to_cart", 3)
        .otherwise(0)
    )

# Get Purchases from Orders (Score = 5)
purchases = orders.join(items, "order_id") \
    .join(customers, "customer_id") \
    .select(col("customer_unique_id"), col("product_id")) \
    .withColumn("score", lit(5)) \
    .withColumn("event_type", lit("purchase"))

# Combine
interaction_matrix = events.unionByName(purchases, allowMissingColumns=True) \
    .groupBy("customer_unique_id", "product_id") \
    .agg(sum("score").alias("interaction_strength"))

# Save to Gold
interaction_matrix.write.format("delta").mode("overwrite").saveAsTable("ecommerce_gold.interaction_matrix")

print("✅ Interaction Matrix created.")

# COMMAND ----------

