# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER
# MAGIC # THERE IS A 2 POINT DEDUCTION IF YOU RUN ANYTHING IN THE SHARED FOLDER. THANKS!

# COMMAND ----------

3month_cleaned.columns

# COMMAND ----------

from pyspark.sql.functions import col
print("Welcome to the W261 final project!") 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Know your mount
# MAGIC Here is the mounting for this class, your source for the original data! Remember, you only have Read access, not Write! Also, become familiar with `dbutils` the equivalent of `gcp` in DataProc

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data for the Project
# MAGIC
# MAGIC For the project you will have 4 sources of data:
# MAGIC
# MAGIC 1. Airlines Data: This is the raw data of flights information. You have 3 months, 6 months, 1 year, and full data from 2015 to 2019. Remember the maxima: "Test, Test, Test", so a lot of testing in smaller samples before scaling up! Location of the data? `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/`, `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/`, etc. (Below the dbutils to get the folders)
# MAGIC 2. Weather Data: Raw data for weather information. Same as before, we are sharing 3 months, 6 months, 1 year
# MAGIC 3. Stations data: Extra information of the location of the different weather stations. Location `dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/`
# MAGIC 4. OTPW Data: This is our joined data (We joined Airlines and Weather). This is the main dataset for your project, the previous 3 are given for reference. You can attempt your own join for Extra Credit. Location `dbfs:/mnt/mids-w261/OTPW_60M/` and more, several samples are given!

# COMMAND ----------

# Airline Data    
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/")
display(df_flights)

# COMMAND ----------

# Weather data
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
display(df_weather)

# COMMAND ----------

# Stations data      
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
display(df_stations)

# COMMAND ----------

# OTPW
df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
display(df_otpw)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Example of EDA

# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import

# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt

#df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/")
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")

# Grouping and aggregation for df_stations
grouped_stations = df_stations.groupBy('neighbor_id').agg(
    F.avg('distance_to_neighbor').alias('avg_distance_to_neighbor'),
).orderBy('avg_distance_to_neighbor')

display(grouped_stations)

# Grouping and aggregation for df_flights
grouped_flights = df_flights.groupBy('OP_UNIQUE_CARRIER').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY'),
    F.avg('DISTANCE').alias('Avg_DISTANCE')
)

display(grouped_flights)

# Convert columns to appropriate data types
df_weather = df_weather.withColumn("HourlyPrecipitationDouble", F.col("HourlyPrecipitation").cast("double"))
df_weather = df_weather.withColumn("HourlyVisibilityDouble", F.col("HourlyVisibility").cast("double"))
df_weather = df_weather.withColumn("HourlyWindSpeedDouble", F.col("HourlyWindSpeed").cast("double")).filter(col("HourlyWindSpeedDouble") < 2000)

# Overlayed boxplots for df_weather
weather_cols = ['HourlyPrecipitationDouble', 'HourlyVisibilityDouble', 'HourlyWindSpeedDouble']
weather_data = df_weather.select(*weather_cols).toPandas()

#plt.figure(figsize=(10, 6))
#weather_data.boxplot(column=weather_cols)
#plt.title('Boxplots of Weather Variables')
#plt.xlabel('Weather Variables')
#plt.ylabel('Values')
#plt.xticks(rotation=45)
#plt.show()

display(df_weather.select(*weather_cols))


# COMMAND ----------

# MAGIC %md 
# MAGIC # Pipeline Steps For Classification Problem
# MAGIC
# MAGIC These are the "normal" steps for a Classification Pipeline! Of course, you can try more!
# MAGIC
# MAGIC ## 1. Data cleaning and preprocessing
# MAGIC
# MAGIC * Remove outliers or missing values
# MAGIC * Encode categorical features
# MAGIC * Scale numerical features
# MAGIC
# MAGIC ## 2. Feature selection
# MAGIC
# MAGIC * Select the most important features for the model
# MAGIC * Use univariate feature selection, recursive feature elimination, or random forest feature importance
# MAGIC
# MAGIC ## 3. Model training
# MAGIC
# MAGIC * Train a machine learning model to predict delays more than 15 minutes
# MAGIC * Use logistic regression, decision trees, random forests, or support vector machines
# MAGIC
# MAGIC ## 4. Model evaluation
# MAGIC
# MAGIC * Evaluate the performance of the trained model on a holdout dataset
# MAGIC * Use accuracy, precision, recall, or F1 score
# MAGIC
# MAGIC ## 5. Model deployment
# MAGIC
# MAGIC * Deploy the trained model to a production environment
# MAGIC * Deploy the model as a web service or as a mobile app
# MAGIC
# MAGIC ## Tools
# MAGIC
# MAGIC * Spark's MLlib and SparkML libraries
# MAGIC * These libraries have parallelized methods for data cleaning and preprocessing, feature selection, model training, model evaluation, and model deployment which we will utilize for this classification problem.
# MAGIC

# COMMAND ----------

df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
display(df_otpw)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, when, concat, lag, substring
from pyspark.sql.window import Window

# COMMAND ----------

## Flight Data EDA
### Flight Data
## test df
df = df_otpw
### filter null delay flights
df = df.filter(df.DEP_DELAY_NEW.isNotNull())
### Filter Non cancelled flights
### Create flight ID (mix of Tail ID + Airline + Date)
df = df.withColumn("flight_id", concat(col("OP_CARRIER_FL_NUM"), col("ORIGIN_AIRPORT_ID"), col("DEST_AIRPORT_ID"), col("OP_CARRIER_AIRLINE_ID"), col("MONTH"), col("DAY_OF_MONTH"), col("YEAR")))



# COMMAND ----------

## Create Time based variable

### Create variable to see if tail num is in the air 2 hrs before departure time
windowSpec = Window.partitionBy("TAIL_NUM").orderBy("YEAR", "MONTH", "DAY_OF_MONTH", "DEP_TIME")

# Define a lag window function to look back 2 hours
lag_window_DoM = lag(col("DAY_OF_MONTH"), 1).over(windowSpec)
lag_window_dep = lag(col("DEP_TIME"), 1).over(windowSpec)
lag_window_arr = lag(col("ARR_TIME"), 1).over(windowSpec)
lag_window_delay = lag(col("DEP_DEL15"), 1).over(windowSpec)

# Check if plane is in the air & if it is delayed
df_stg = df.withColumn(
   "is_tail_num_in_air_prev_2_hrs",
   when(
      (
         (F.col("CRS_DEP_TIME").cast("double") - 200) > lag_window_dep.cast("double")
      )
      &
      (
         (F.col("CRS_DEP_TIME").cast("double") - 200) < lag_window_arr.cast("double")
      )
      &
      (lag_window_DoM == F.col("DAY_OF_MONTH")),
      1
   ).otherwise(0)
)
df = df_stg.withColumn(
    'is_prev_flight_delayed',
    when((lag_window_delay.cast("double") == 1)
               &
      (lag_window_DoM == F.col("DAY_OF_MONTH")), 1).otherwise(0)

)

# Show the resulting DataFrame
display(df)


# COMMAND ----------

## Place this cell in any team notebook that needs access to the team cloud storage.


# The following blob storage is accessible to team members only (read and write)
# access key is valid til TTL
# after that you will need to create a new SAS key and authenticate access again via DataBrick command line
blob_container  = "data"       # The name of your container created in https://portal.azure.com
storage_account = "rodrigocochran" # The name of your Storage account created in https://portal.azure.com
secret_scope    = "team_2_1"            # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "team_2_1_key"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)



# see what's in the blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# Loading Pagerank
df_otpw_60m = spark.read.parquet(f"{team_blob_url}/60month_cleaned_new_1")

df_otpw_60m_2015 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2015)
df_otpw_60m_2016 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2016)
df_otpw_60m_2017 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2017)
df_otpw_60m_2018 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2018) 

pageranks_2015 = spark.read.parquet(f"{team_blob_url}/pageranks/_2015")
pageranks_2016 = spark.read.parquet(f"{team_blob_url}/pageranks/_2016")
pageranks_2017 = spark.read.parquet(f"{team_blob_url}/pageranks/_2017")
pageranks_2018 = spark.read.parquet(f"{team_blob_url}/pageranks/_2018")

# Example of how to join it to the otpw df.
df_otpw_60m_with_ranks = df_otpw_60m_2018.join(other=pageranks_2018,
                                         on= df_otpw_60m_2018.ORIGIN_AIRPORT_ID == pageranks_2018.id , 
                                         how='left')

# COMMAND ----------

##### Delays by DoW
delay_by_DoW = df_otpw_60m_with_ranks.groupby("DAY_OF_WEEK").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
display(delay_by_DoW)

# COMMAND ----------

## Delay by Day of Week Graph
vals = delay_by_DoW.select(delay_by_DoW[1]).collect()
DoW = delay_by_DoW.select(delay_by_DoW[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
DoW = [d[0] for d in DoW]
plt.bar(DoW, vals, color='skyblue')
plt.xlabel('Day of Week')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by DoW")
display(plt.show())

# COMMAND ----------

##### Delays by Airline
delay_by_airline = df_otpw_60m_with_ranks.groupby("OP_UNIQUE_CARRIER").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
display(delay_by_airline)

# COMMAND ----------

## Delay by Carrier List
vals = delay_by_airline.select(delay_by_airline[1]).collect()
carr = delay_by_airline.select(delay_by_airline[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
carr = [c[0] for c in carr]
plt.bar(carr, vals, color='green')
plt.xlabel('Carrier')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_Carrier")
display(plt.show())

# COMMAND ----------

##### Delays by ARRIVAL Airport
delay_by_arr = df.groupby("DEST").agg(F.avg("DEP_DEL15").alias('Delay_pct'),
F.count('flight_id').alias('num_flights')).orderBy(F.col("Delay_pct").desc())
display(delay_by_arr)

# COMMAND ----------

## Delay by Arrival
vals = delay_by_arr.select(delay_by_arr[1]).collect()
arr = delay_by_arr.select(delay_by_arr[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
arr = [a[0] for a in arr]
## Get Top and Bottom 10 vals
# Select top 10 highest values
top_10_highest_d = arr[:10]
top_10_highest_dr = vals[:10]
# Select top 10 lowest values
top_10_lowest_d = arr[-10:]
top_10_lowest_dr = vals[-10:]
# Plot the top 10 highest values
plt.subplot(2, 1, 1)
plt.bar(top_10_highest_d, top_10_highest_dr, color='purple')
plt.xlabel('Arrival Airport')
plt.ylabel('Average Delay Rate')
plt.title('Top 10 Highest Average Delay Rate by Arrival Airport')

# Plot the top 10 lowest values
plt.subplot(2, 1, 2)
plt.bar(top_10_lowest_d, top_10_lowest_dr, color='orange')
plt.xlabel('Arrival Airport')
plt.ylabel('Average Delay Rate')
plt.title('Top 10 Lowest Average Delay Rate by Arrival Airport')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
display(plt.show())

# COMMAND ----------

##### Delays by DEPARTURE Airport
delay_by_dept = df.groupby("ORIGIN").agg(F.avg("DEP_DEL15").alias('Delay_pct'),
F.count('flight_id').alias('num_flights')).orderBy(F.col("Delay_pct").desc())
display(delay_by_dept)

# COMMAND ----------

## Delays by Departure
vals = delay_by_dept.select(delay_by_dept[1]).collect()
dept = delay_by_dept.select(delay_by_dept[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
dept = [d[0] for d in dept]
## Get Top and Bottom 10 vals
# Select top 10 highest values
top_10_highest_d = dept[:10]
top_10_highest_dr = vals[:10]
# Select top 10 lowest values
top_10_lowest_d = dept[-10:]
top_10_lowest_dr = vals[-10:]
# Plot the top 10 highest values
plt.subplot(2, 1, 1)
plt.bar(top_10_highest_d, top_10_highest_dr, color='red')
plt.xlabel('Departure Airport')
plt.ylabel('Average Delay Rate')
plt.title('Top 10 Highest Average Delay Rate by Departure Airport')

# Plot the top 10 lowest values
plt.subplot(2, 1, 2)
plt.bar(top_10_lowest_d, top_10_lowest_dr, color='lightcoral')
plt.xlabel('Departure Airport')
plt.ylabel('Average Delay Rate')
plt.title('Top 10 Lowest Average Delay Rate by Departure Airport')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
display(plt.show())

# COMMAND ----------

##### Delays by DISTANCE group
delay_by_dist = df_otpw_60m_with_ranks.groupby("DISTANCE_GROUP").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
display(delay_by_dist)

# COMMAND ----------

## Delays by Distance Group
vals = delay_by_dist.select(delay_by_dist[1]).collect()
dist = delay_by_dist.select(delay_by_dist[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
dist = [d[0] for d in dist]
plt.bar(dist, vals, color='yellow')
plt.xlabel('Distance')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_Distance")
display(plt.show())

# COMMAND ----------

##### Delays by TIME
delay_by_time = df_otpw_60m_with_ranks.groupby(F.substring(df_otpw_60m_with_ranks["CRS_DEP_TIME"], 1, 2).alias("DEP_TIME_BLK_SUBSTR")) \
    .agg(F.avg(col("DEP_DEL15")).alias('Delay_pct')) \
    .orderBy("Delay_pct")

display(delay_by_time)

# COMMAND ----------

## Delays by Time Group
vals = delay_by_time.select(delay_by_time[1]).collect()
time = delay_by_time.select(delay_by_time[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
time = [t[0] for t in time]
plt.bar(time, vals, color='black')
plt.xlabel('Time')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_Time")
display(plt.show())

# COMMAND ----------

# No Need to create convert to numeric field since all non-numeric field have an id equivalent
## Convert key non-numeric fields to numeric
### Get unique values in a list
# airlines = df.select("OP_UNIQUE_CARRIER").distinct().collect() 
# airline_list = [row.OP_UNIQUE_CARRIER for row in airlines]
# time_block = df.select("DEP_TIME_BLK").distinct().collect() 
# time_block_list = [row.DEP_TIME_BLK for row in time_block]
# airports = df.select("ORIGIN").distinct().collect() 
# airport_list = [row.ORIGIN for row in airports]


# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import NumericType
from pyspark.sql.functions import col
import pandas as pd

# COMMAND ----------

## Delete info known only after delay

### Delete Arrival related info
df = df.drop('TAXI_IN',
'CRS_ARR_TIME',
'ARR_TIME',
'ARR_DELAY',
'ARR_DELAY_NEW',
'ARR_DEL15',
'ARR_DELAY_GROUP',
'CRS_ELAPSED_TIME',
'ACTUAL_ELAPSED_TIME',
'AIR_TIME',
'WHEELS_ON')

### Delete Delay related info (all variables besides output)
df = df.drop(
'CARRIER_DELAY',
'WEATHER_DELAY',
'NAS_DELAY',
'SECURITY_DELAY',
'LATE_AIRCRAFT_DELAY',
'FIRST_DEP_TIME',
'TOTAL_ADD_GTIME',
'DEP_TIME',
'DEP_DELAY',
'DEP_DELAY_NEW',
'TAXI_OUT',
'WHEELS_OFF')

# COMMAND ----------

## Convert cols to numeric
for column_name in df.columns:
    df = df.withColumn(column_name, col(column_name).cast("double"))
display(df)

# COMMAND ----------

## Drop non-numeric fields
# List of columns with non-numeric values
non_numeric_cols = []

# Iterate over the columns and check the data type
for column_name in df.columns:
    data_type = df.schema[column_name].dataType
    if not isinstance(data_type, NumericType):
        non_numeric_cols.append(column_name)

# Drop columns with non-numeric values
df = df.drop(*non_numeric_cols)

## Drop Null columns
df = df.dropna(how="all")

## Drop Leftover Columns
df = df.drop('FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER', 'TAIL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_NM', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'CANCELLATION_CODE')

## Drop Weather Features (Flight Data only)

# Get the index of the specified column
months_index = df.columns.index("MONTH")
# flight_id_index = df.columns.index("flight_id")

# Drop columns to the right of the specified column
columns_to_drop = df.columns[months_index+ 1:]
df = df.drop(*columns_to_drop)

## Add Flight ID
df = df.withColumn("flight_id", concat(col("OP_CARRIER_FL_NUM"), col("ORIGIN_AIRPORT_ID"), col("DEST_AIRPORT_ID"), col("OP_CARRIER_AIRLINE_ID"), col("MONTH"), col("DAY_OF_MONTH"), col("YEAR")))
df = df.withColumn("flight_id", col("flight_id").cast("double"))

display(df)

# COMMAND ----------

df = df.drop('flight_id')
df = df.na.drop()

# COMMAND ----------

## Show Correlation Matrix
# Assemble the feature vector
vector_assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
assembled_df = vector_assembler.transform(df)

# Calculate correlation matrix
correlation_matrix = Correlation.corr(assembled_df, "features").head()

# Extract the correlation matrix as a dense matrix
corr_matrix = correlation_matrix[0].toArray()

# Get the list of feature names
feature_names = df.columns

# Convert the correlation matrix to a DataFrame with feature names as index and columns
corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)

# Display the correlation matrix DataFrame
print("Correlation Matrix:")
print(corr_df)

corr_df.to_csv("correlation_matrix.csv")

## Redundant Cols --> ORIGIN_AIRPORT_SEQ_ID, DEST_AIRPORT_SEQ_ID, DEP_DELAY_NEW, ARR_DELAY_NEW, LONGEST_ADD_GTIME

## Most Correlated Features to DEP_DELAY_15 --> DEP_DELAY_GROUP, ARR_DELAY_GROUP_, TOTAL_ADD_GTIME, LONGEST_ADD_GTIME

# COMMAND ----------

## Remove highly correlated features
df.drop('ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW', 'LONGEST_ADD_GTIME')


# COMMAND ----------



# COMMAND ----------


