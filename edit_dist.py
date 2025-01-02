import os
from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql.types import IntegerType
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from pyspark.sql import functions as F

def edit_distance(pair):
    str1, str2 = pair
    # Create a table to store results of subproblems
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill dp table
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # Min operations = j
            elif j == 0:
                dp[i][j] = i  # Min operations = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    # Insert
                                   dp[i - 1][j],    # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]

# Function to compute edit distances using multiple processes
def compute_edit_distance_multiprocess(pair, num_workers):
    # implement a multi-process version of edit_distance function
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(edit_distance, pair)
    return results
    #raise NotImplementedError("This function is not implemented yet.")


if __name__=="__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, default='simple-wiki-unique-has-end-punct-sentences.csv', help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, default=300, help="Number of sentences")
    args = parser.parse_args()

    # Number of processes to use (set to the number of CPU cores)
    num_workers = multiprocessing.cpu_count()
    # print(f'number of available cpu cores: {num_workers}')
    # Sample list of string pairs
    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EditDistanceComputation") \
        .getOrCreate()
    # Convert pair data to Spark DataFrame
    pair_df = spark.createDataFrame(pair_data, schema=["sentence1", "sentence2"])
    # Register the edit distance function as a pandas UDF for single pairs
    @pandas_udf(IntegerType())
    def edit_distance_udf(str1: pd.Series, str2: pd.Series) -> pd.Series:
        return str1.combine(str2, lambda x, y: edit_distance((x, y)))
    # Compute edit distances using Spark
    start_time = time.time()
    result_df = pair_df.withColumn("edit_distance", edit_distance_udf(col("sentence1"), col("sentence2")))
    # result_df.select(F.count("*")).show()
    end_time = time.time()
    spark_time = end_time - start_time
    # print(f"Time taken (Spark): {end_time - start_time:.2f} seconds")

    # Stop the Spark session
    spark.stop()

    #Multiprocess
    start_time = time.time()
    # Compute edit distances in parallel
    edit_distances = compute_edit_distance_multiprocess(pair_data, num_workers)
    end_time = time.time()
    # print(f"Number of string pairs processed: {len(edit_distances)}")
    # print(f"Time taken (multi-process): {end_time - start_time:.3f} seconds")
    multi_process_time = end_time - start_time
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # For loop

    start_time = time.time()
    distances = []
    for pair in tqdm(pair_data, ncols=100):
        distances.append(edit_distance(pair))
    end_time = time.time()
    # print(f"Edit Distances (Non-Spark): {distances}")
    # print(f"Time taken (for-loop): {end_time - start_time:.3f} seconds")
    for_loop_time = end_time - start_time
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f"Time cost (Spark, multi-process, for-loop): [{spark_time:.3f}, {multi_process_time:.3f}, {for_loop_time:.3f}]")