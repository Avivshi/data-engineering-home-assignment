from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, avg, stddev, when
from pyspark.sql.window import Window


BUCKET = "s3://data-engineer-assignment-aviv-shimoni"


def load_data(spark, file_path):
    """Load data into a DataFrame."""
    return spark.read.csv(file_path, header=True, inferSchema=True)


def calculate_daily_returns(df):
    """Calculate daily returns using closing prices."""
    window_spec = Window.partitionBy("ticker").orderBy("Date")
    df = df.withColumn("prev_close", lag("close").over(window_spec))
    df = df.withColumn("daily_return",
                       when(col("prev_close").isNotNull(),
                            (col("close") - col("prev_close")) / col("prev_close")).otherwise(0))
    return df


def compute_average_daily_return(df):
    """Compute the average daily return of all stocks for every date."""
    return df.groupBy("Date").agg(avg("daily_return").alias("average_return"))


def calculate_average_worth(df):
    """Calculate worth and average worth per ticker, and identify the ticker with the highest average worth."""
    df = df.withColumn("worth", col("close") * col("volume"))
    avg_worth = df.groupBy("ticker").agg(avg("worth").alias("value"))
    max_avg_worth = avg_worth.orderBy(col("value").desc()).limit(1)
    return max_avg_worth


def calculate_most_volatile_stock(df):
    """Identify the most volatile stock based on the standard deviation of daily returns."""
    volatility = df.groupBy("ticker").agg(stddev("daily_return").alias("standard_deviation"))
    most_volatile = volatility.orderBy(col("standard_deviation").desc()).limit(1)
    return most_volatile


def calculate_top_30_day_returns(df):
    """Identify the top three 30-day return dates for stocks."""
    window_spec = Window.partitionBy("ticker").orderBy("Date")
    df = df.withColumn("prev_30_close", lag("close", 30).over(window_spec))
    df = df.withColumn("return_30_day",
                       when(col("prev_30_close").isNotNull(),
                            (col("close") - col("prev_30_close")) / col("prev_30_close")).otherwise(0))
    return df.orderBy(col("return_30_day").desc()).select("ticker", "Date").limit(3)


def save_dataframe(df, file_name):
    """Save DataFrame to S3."""
    output_path_s3 = f"{BUCKET}/results/{file_name}/"
    df.write.mode("overwrite").csv(output_path_s3, header=True)  # Save to S3


def main():
    spark = SparkSession.builder.appName("Stock Analysis").getOrCreate()
    df = spark.read.csv(f"{BUCKET}/data/stocks_data.csv", header=True, inferSchema=True)

    # Calculate daily returns
    df = calculate_daily_returns(df=df)

    # Objective 1: Average Daily Return
    avg_daily_return = compute_average_daily_return(df=df)
    save_dataframe(df=avg_daily_return, file_name="average_daily_return")

    # Objective 2: Highest Average Worth
    highest_worth_ticker = calculate_average_worth(df=df)
    save_dataframe(df=highest_worth_ticker, file_name="highest_worth")

    # Objective 3: Most Volatile Stock
    most_volatile_stock = calculate_most_volatile_stock(df=df)
    save_dataframe(df=most_volatile_stock, file_name="most_volatile")

    # Objective 4: Top Three 30-Day Return Dates
    top_30_day_returns = calculate_top_30_day_returns(df=df)
    save_dataframe(df=top_30_day_returns, file_name="top_30_day_returns")

    spark.stop()


if __name__ == "__main__":
    main()
