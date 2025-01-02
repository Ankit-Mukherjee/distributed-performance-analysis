import os
import warnings
warnings.filterwarnings("ignore")
os.system('clear')
import argparse
import torch
import torch.nn as nn
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf,explode
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims):
        super(MLPClassifier, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_classes))  # Output layer with `num_classes` units
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Compute logits
        logits = self.model(x)
        # Get predicted class using argmax
        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes

# Define a pandas UDF for parallel classification
@pandas_udf(ArrayType(IntegerType()))
def MLPClassifier_udf(*batch_inputs):
    batch_tensor = torch.tensor(batch_inputs, dtype=torch.float32).T  
    model.eval()
    with torch.no_grad():
        predictions = model(batch_tensor)  
    return pd.Series(predictions.tolist())  

if __name__=="__main__":
    # Set up argument parsing
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of sentences")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="hidden_dim")
    parser.add_argument('--hidden_layer', type=int, default=50, help="hidden_layer")
    args = parser.parse_args()

    # Configuration
    input_dim = 128  # Input dimension
    num_classes = 10  # Number of classes
    hidden_dims = [args.hidden_dim * args.hidden_layer]  # Hidden layer sizes
    # Model and input setup
    mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)
    x = torch.randn(args.n_input, input_dim)  # A random input vector of dimension n
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Spark version

    # Initialize Spark session
    
    spark = SparkSession.builder \
        .appName("MLPClassifier") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Convert Pandas DataFrame to Spark DataFrame
    input_df = pd.DataFrame(x.numpy())
    spark_df = spark.createDataFrame(input_df)

    # Apply the UDF to perform distributed classification
    start_time = time.time()
    result_df = spark_df.withColumn("predictions", MLPClassifier_udf(*[spark_df[c] for c in spark_df.columns]))
    result_df = result_df.withColumn("prediction", explode(result_df.predictions))

    end_time = time.time()
    time_1 = end_time - start_time
    # print(f"Time taken for distributed classification: {end_time - start_time:.6f} seconds")

    # Stop Spark session
    spark.stop()


    # Timing the forward pass
    start_time = time.time()
    output = mlp_model(x)
    end_time = time.time()

    # Output and timing results
    time_2 = end_time - start_time
    # print(f"Output: {output.shape}")
    # print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
    print(f"Time cost for spark and non-spark version: [{time_1:.3f},  {time_2:.3f}] seconds")

