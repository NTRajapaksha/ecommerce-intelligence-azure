import pandas as pd
import numpy as np
from faker import Faker
import os
import random
from datetime import timedelta
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# 1. Load Environment Variables
load_dotenv()

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "./raw_data")
STORAGE_CONN_STR = f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_KEY')};EndpointSuffix=core.windows.net"
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "bronze")

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)

os.makedirs(DATA_DIR, exist_ok=True)

def upload_to_azure(file_path, blob_name):
    """
    Uploads a local file to Azure Blob Storage (Bronze Layer)
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        
        # Create container if it doesn't exist
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container()
            print(f"Created container: {CONTAINER_NAME}")

        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        print(f"Uploading {blob_name} to Azure...")
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"✅ Uploaded {blob_name} successfully.")
        
    except Exception as e:
        print(f"❌ Failed to upload {blob_name}: {e}")

def load_olist_data():
    """
    Checks for Olist data in DATA_DIR.
    """
    print("Checking for Olist Orders...")
    try:
        # We only need a few files to generate synthetic links
        orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
        customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
        products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
        return orders, customers, products
    except FileNotFoundError:
        print(f"⚠️  Olist datasets not found in {DATA_DIR}.") 
        print("Please download 'Brazilian E-Commerce Public Dataset by Olist' from Kaggle.")
        return None, None, None

def generate_clickstream(orders_df, customers_df, products_df, n_events=50000):
    print(f"Generating {n_events} synthetic clickstream events...")
    
    events = []
    real_order_sample = orders_df.sample(n=min(len(orders_df), int(n_events * 0.2))) 
    
    customer_ids = customers_df['customer_unique_id'].unique()
    product_ids = products_df['product_id'].unique()
    event_types = ['page_view', 'product_view', 'add_to_cart', 'checkout_start']
    weights = [0.6, 0.25, 0.1, 0.05]
    
    # 1. Generate events linked to real purchases
    for _, row in real_order_sample.iterrows():
        purchase_time = pd.to_datetime(row['order_purchase_timestamp'])
        session_id = fake.uuid4()
        
        # Try to find the unique ID, fallback if data mismatch
        cust_row = customers_df[customers_df['customer_id'] == row['customer_id']]
        if not cust_row.empty:
            cust_id = cust_row['customer_unique_id'].values[0]
        else:
            cust_id = fake.uuid4()

        # Generate funnel
        num_events = random.randint(3, 8)
        for i in range(num_events):
            time_offset = random.randint(5, 60)
            event_time = purchase_time - timedelta(minutes=time_offset - i)
            
            events.append({
                'event_id': fake.uuid4(),
                'session_id': session_id,
                'customer_unique_id': cust_id,
                'event_type': np.random.choice(event_types, p=weights),
                'product_id': np.random.choice(product_ids) if random.random() > 0.3 else None,
                'timestamp': event_time,
                'device': np.random.choice(['mobile', 'desktop'], p=[0.65, 0.35]),
                'location_state': 'SP' # Simplified
            })

    # 2. Random Noise
    remaining_events = n_events - len(events)
    if remaining_events > 0:
        for _ in range(remaining_events):
            events.append({
                'event_id': fake.uuid4(),
                'session_id': fake.uuid4(),
                'customer_unique_id': np.random.choice(customer_ids) if random.random() > 0.5 else fake.uuid4(),
                'event_type': np.random.choice(event_types, p=weights),
                'product_id': np.random.choice(product_ids) if random.random() > 0.3 else None,
                'timestamp': pd.to_datetime('2018-01-01') + timedelta(days=random.randint(0, 300)),
                'device': np.random.choice(['mobile', 'desktop'], p=[0.65, 0.35]),
                'location_state': np.random.choice(['SP', 'RJ', 'MG'])
            })
            
    return pd.DataFrame(events)

if __name__ == "__main__":
    orders, customers, products = load_olist_data()
    
    if orders is not None:
        # Generate
        df_clickstream = generate_clickstream(orders, customers, products, n_events=50000)
        
        # Save Local
        local_path = f"{DATA_DIR}/synthetic_clickstream.csv"
        df_clickstream.to_csv(local_path, index=False)
        print(f"Saved locally to {local_path}")
        
        # Upload to Azure
        if os.getenv('AZURE_STORAGE_KEY'):
            upload_to_azure(local_path, "synthetic_clickstream.csv")
            
            # Also upload the Olist files if they exist
            upload_to_azure(f"{DATA_DIR}/olist_orders_dataset.csv", "olist_orders_dataset.csv")
            upload_to_azure(f"{DATA_DIR}/olist_customers_dataset.csv", "olist_customers_dataset.csv")
            upload_to_azure(f"{DATA_DIR}/olist_products_dataset.csv", "olist_products_dataset.csv")
            upload_to_azure(f"{DATA_DIR}/olist_order_items_dataset.csv", "olist_order_items_dataset.csv")
        else:
            print("⚠️ Skipped Azure Upload: AZURE_STORAGE_KEY not found in .env")