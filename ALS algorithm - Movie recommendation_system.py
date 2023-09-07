from pyspark.sql.functions import lit
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import datetime

# Initialize spark session
spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "15g") \
    .appName("MovieRecommender") \
    .config("spark.logConf", "true") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/MovieLens") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/MovieLens") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.0")\
    .getOrCreate()

def get_rmse(predictions):
    """ Return the RMSE of the predicted model """
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return rmse

def get_dataframe (table):
    """Returns a collection from mongo as a spark dataframe"""
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("collection", table).load()
    df.show(10)
    return df

def insert_dataframe (dataframe,table_name):
    """Save a spark dataframe in a mongo collection"""
    dataframe.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").option('sampleSize', 50000).option("collection", table_name).save()

def quiet_logs(sc):
    """Disables the default verbose logging of spark"""
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )

def print_current_time(message):
    """Prints the current time for logging purposes"""
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    if (now.hour < 10):
        hour = f"0{now.hour}"
    if (now.minute < 10):
        minute = f"0{now.minute}"
    print(f"{message} {hour}:{minute}")

def predict (sample_list,personal_id):
    """Generates a prediction model"""

    print_current_time("Program started at")

    sc = spark.sparkContext
    sc.setLogLevel("FATAL")

    # Let's make a dataframe from our data, imported as a json (useful if the request is coming from an HTTP request)
    rdd = sc.parallelize([sample_list])
    data_me = spark.read.json(rdd).select("movieId","userId","rating")

    quiet_logs(sc)

    # Let's load movies and ratings
    movies = get_dataframe("movies_typed")
    ratings = get_dataframe("ratings_typed")

    # Here we want to filter movies with less than 1000 reviews, in order to get recommendations about sufficiently known movies
    counter = ratings.groupBy("movieId").count()
    high_rated_movies = counter.filter((counter["count"] > 1000)).orderBy("count",ascending=False)

    if (len(high_rated_movies.head(1)) == 0):
        return

    ratings = ratings.join(high_rated_movies,['movieId']).join(movies,['movieId']).select(['movieId','userId','rating'])
    ratings = ratings.union(data_me)

    # Let's start the training
    (training, test) = ratings.randomSplit([0.7, 0.3])
    data_left = movies.select('movieId').withColumn('userId',lit(personal_id))
    print("Training started")
    als = ALS(maxIter=10, regParam=0.06, rank=30, userCol="userId",itemCol = "movieId", ratingCol = "rating", coldStartStrategy = "drop")
    model = als.fit(training)

    # Here we are going to set predictions about our personal tastes
    predictions = model.transform(data_left).join(movies.select(["movieId","title"]),["movieId"]).orderBy("prediction",ascending=False)
    predictions.show(20)

    # Here we are going to set prediction for a small test sample
    predictions_test = model.transform(test).join(movies.select(["movieId","title"]),["movieId"])
    predictions_test.show(20)

    # Inserting back in mongo (notice that mongo does not accept floats, so we have to cast float into doubles)
    insert_dataframe(predictions.withColumn("prediction_double", predictions["prediction"].cast(DoubleType())).select(["movieId","userId","prediction_double","title"]),"predictions")

    # Get the rmse
    print(get_rmse(predictions_test.na.drop()))
    
    print_current_time("Program finished at")

# Personal Id
personalId = 100000

# List of preferences to be processed (mostly action movies). It is build as a python dictionary/JSON, so this way it can also passed by an HTML POST request
preferences = [
    {"movieId": 2, "userId": personalId, "rating": 5},
    {"movieId": 260, "userId": personalId, "rating": 5},
    {"movieId": 783, "userId": personalId, "rating": 4},
    {"movieId": 1562, "userId": personalId, "rating": 3},
    {"movieId": 5349, "userId": personalId, "rating": 5}
]

# Execute the main process
predict(preferences,personalId)