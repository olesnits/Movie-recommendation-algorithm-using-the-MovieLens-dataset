from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
from pymongo import MongoClient

spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "15g") \
    .appName("MovieRecommender") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/MovieLens") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/MovieLens") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.0")\
    .getOrCreate()

def get_rmse(predictions):
    """ Return the RMSE of the predicted model """
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return rmse

sc = spark.sparkContext

# Extracting the tables from database
movies = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("collection", "movies_typed").load()
ratings = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("collection", "ratings_typed").load()
data = movies.join(ratings, ['movieId']).select(['movieId','title','genres','userId','rating'])

# Filtering the games with less than 1000 ratings
counter = data.groupBy("movieId","genres").count()
high_rated_movies = counter.filter(counter["count"] > 1000).orderBy("count",ascending=False)
high_rated_movies.show(10)

# Extracting a training and a testing sample
(training, test) = data.randomSplit([0.7, 0.3])

# We are here going to test different values of lambda and latent factors
for regParam in [0.01,0.04,0.06,0.08,0.1,0.3,0.6]:
    for hidden in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]: # After 25 it is all the same, so we are going to take 25
            als = ALS(maxIter=10, regParam=regParam, rank=hidden, userCol="userId",itemCol = "movieId", ratingCol =    "rating", coldStartStrategy = "drop")
            model = als.fit(training)
            predictions = model.transform(test).orderBy("prediction",ascending=False)
            print(f"RMSE: {get_rmse(predictions)} for regparam={regParam} and hidden={hidden}")
            log_file = open('log.txt', 'a')
            log_file.write(f"{get_rmse(predictions)},{regParam},{hidden}\n")