import json
import numpy as np
import os
import mlflow.spark
from pyspark.sql import SparkSession

def init():
    """
    This function is called when the container starts.
    It loads the model into memory.
    """
    global model
    global spark
    
    # Initialize a lightweight Spark Session (Required for ALS)
    spark = SparkSession.builder.appName("Scoring").getOrCreate()
    
    # Load the model from the Azure ML Registry path
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "als-model")
    model = mlflow.spark.load_model(model_path)
    print("✅ Model Loaded Successfully.")

def run(raw_data):
    """
    This function is called for every API request.
    Input: JSON string '{"user_id": 123}'
    Output: JSON response
    """
    try:
        data = json.loads(raw_data)
        user_int = data['user_id']
        
        # Create a DataFrame for the single user
        df_input = spark.createDataFrame([(user_int,)], ["user_int"])
        
        # Generate Recommendations
        recs = model.recommendForUserSubset(df_input, 5)
        
        # Format Output
        result = recs.collect()[0]['recommendations']
        return json.dumps({"status": "success", "recommendations": result})
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
