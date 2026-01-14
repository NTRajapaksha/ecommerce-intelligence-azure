# Databricks notebook source
# MAGIC %md
# MAGIC Setup Paths & Configuration

# COMMAND ----------

# Configuration
storage_name = "AZURE_STORAGE_ACCOUNT_NAME" 
container_name = "bronze"

# Base Path using ABFSS driver (Standard for Data Lake Gen2)
base_path = f"abfss://{container_name}@{storage_name}.dfs.core.windows.net"

# Define raw file paths
paths = {
    "orders": f"{base_path}/olist_orders_dataset.csv",
    "items": f"{base_path}/olist_order_items_dataset.csv",
    "customers": f"{base_path}/olist_customers_dataset.csv",
    "products": f"{base_path}/olist_products_dataset.csv",
    "clickstream": f"{base_path}/synthetic_clickstream.csv"
}

# Silver Layer Path (We will create a 'silver' directory managed by Delta)
silver_path = f"abfss://silver@{storage_name}.dfs.core.windows.net" # Make sure to create 'silver' container or folder if strict

# COMMAND ----------

# MAGIC %md
# MAGIC Define Schema & Read Function

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp, current_timestamp
from pyspark.sql.types import *

def read_bronze_csv(path):
    return spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(path)

# Load DataFrames
df_orders = read_bronze_csv(paths["orders"])
df_items = read_bronze_csv(paths["items"])
df_customers = read_bronze_csv(paths["customers"])
df_clickstream = read_bronze_csv(paths["clickstream"])

print("Data loaded into DataFrames.")

# COMMAND ----------

# MAGIC %md
# MAGIC Transformation (Cleaning & Type Casting)

# COMMAND ----------

# 1. Clean Orders & Items (Type Casting)
df_orders_clean = df_orders \
    .withColumn("order_purchase_timestamp", to_timestamp(col("order_purchase_timestamp"))) \
    .withColumn("order_approved_at", to_timestamp(col("order_approved_at"))) \
    .withColumn("order_delivered_carrier_date", to_timestamp(col("order_delivered_carrier_date"))) \
    .withColumn("order_delivered_customer_date", to_timestamp(col("order_delivered_customer_date"))) \
    .withColumn("order_estimated_delivery_date", to_timestamp(col("order_estimated_delivery_date")))

df_items_clean = df_items \
    .withColumn("price", col("price").cast("float")) \
    .withColumn("freight_value", col("freight_value").cast("float"))

# 2. Clean Clickstream (The Fix)
# Logic: product_id CAN be null for 'page_view', but MUST NOT be null for 'add_to_cart' or 'product_view'
df_clickstream_clean = df_clickstream \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .filter(
        (col("event_type") == "page_view") | 
        (col("event_type") == "checkout_start") |
        ((col("product_id").isNotNull()) & (col("product_id") != "nan"))
    )

print(f"Data Cleaning Complete. Dropped invalid clickstream rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC Write to Silver (Delta Lake)

# COMMAND ----------

# Database Name
spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_silver")

# Write tables
def write_to_delta(df, table_name, partition_col=None):
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_col:
        writer = writer.partitionBy(partition_col)
    
    writer.saveAsTable(f"ecommerce_silver.{table_name}")
    print(f"✅ Table ecommerce_silver.{table_name} saved.")

# Write operations
write_to_delta(df_orders_clean, "orders", partition_col="order_status")
write_to_delta(df_items_clean, "order_items")
write_to_delta(df_customers, "customers")
write_to_delta(df_clickstream_clean, "clickstream", partition_col="event_type")

print("All data moved to Silver Layer (Delta Tables).")

# COMMAND ----------

# MAGIC %md
# MAGIC Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ecommerce_silver.clickstream LIMIT 10

# COMMAND ----------

