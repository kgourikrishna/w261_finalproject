# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER
# MAGIC # THERE IS A 2 POINT DEDUCTION IF YOU RUN ANYTHING IN THE SHARED FOLDER. THANKS!

# COMMAND ----------

from pyspark.sql.functions import col
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd

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

display(dbutils.fs.ls(f"dbfs:/mnt/mids-w261/OTPW_36M/"))

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

df_otpw_12 = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M_2015.csv.gz")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Example of EDA

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

plt.figure(figsize=(10, 6))
weather_data.boxplot(column=weather_cols)
plt.title('Boxplots of Weather Variables')
plt.xlabel('Weather Variables')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

#display(df_weather.select(*weather_cols))


# COMMAND ----------

import pyspark.sql.functions as F

dep_delay = df_flights.select(F.log(df_flights.DEP_DELAY)).alias('Log_Dep_Delay').toPandas()
arr_delay = df_flights.select(F.log(df_flights.ARR_DELAY)).alias('Log_Arr_Delay').toPandas()

fig, axs = plt.subplots(2, 2, figsize=(20, 10))

axs[0,0].hist(dep_delay, bins = 20)
axs[0,1].hist(arr_delay, bins = 20)
dep_delay.boxplot(column = 'ln(DEP_DELAY)', ax = axs[1,0])
arr_delay.boxplot(column = 'ln(ARR_DELAY)', ax = axs[1,1])
fig.suptitle('Log of Arrival and Departure Delays')
axs[0,0].title.set_text('Log Depature Delay')
axs[0,1].title.set_text('Log Arrival Delay')
plt.show()

# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd


grouped_month_avg_delay = df_flights.groupBy('MONTH').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY'),
    F.avg('WEATHER_DELAY').alias('Avg_WEATHER_DELAY'),
    F.avg('CARRIER_DELAY').alias('Avg_CARRIER_DELAY'),
    F.avg('NAS_DELAY').alias('Avg_NAS_DELAY'),
    F.avg('SECURITY_DELAY').alias('Avg_SECURITY_DELAY'),
    F.avg('LATE_AIRCRAFT_DELAY').alias('Avg_LATE_AIRCRAFT_DELAY')
)
grouped_day_avg_delay = df_flights.groupBy('FL_DATE').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY')
)
grouped_dow_avg_delay = df_flights.groupBy('DAY_OF_WEEK').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY')
)
grouped_origin_avg_delay = df_flights.groupBy('ORIGIN').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY')
)

grouped_dest_avg_delay = df_flights.groupBy('DEST').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY'),

)

display(grouped_day_avg_delay)
display(grouped_month_avg_delay)
display(grouped_origin_avg_delay)
display(grouped_dest_avg_delay)



grouped_day_avg_delay_pd  = grouped_day_avg_delay.toPandas()
grouped_dow_avg_delay_pd = grouped_dow_avg_delay.toPandas()


# plt.bar(grouped_dow_avg_delay_pd['DAY_OF_WEEK'], grouped_dow_avg_delay_pd['Avg_ARR_DELAY'])
# plt.show()


# COMMAND ----------

##Weather Dataset EDA

df_weather = df_weather.withColumn('date_only', F.to_date(F.col('DATE')))
df_weather = df_weather.withColumn('MONTH', F.month(F.col('date_only')))
df_weather = df_weather.withColumn('WEEK_NUMBER', F.weekofyear(F.col('date_only')))

weather_group_day = df_weather.groupBy('date_only').agg(
    F.avg("HourlyPrecipitation").alias('Avg_DailyPrecipitation'),
    F.avg('HourlyVisibility').alias('Avg_DailyVisibility'),
    F.avg('HourlyWindSpeed').alias('Avg_DailyWindSpeed')
)
weather_group_week = df_weather.groupBy('WEEK_NUMBER').agg(
    F.avg("HourlyPrecipitation").alias('Avg_WeekPrecipitation'),
    F.avg('HourlyVisibility').alias('Avg_WeekVisibility'),
    F.avg('HourlyWindSpeed').alias('Avg_WeekWindSpeed')
)
weather_group_month = df_weather.groupBy('MONTH').agg(
    F.avg("HourlyPrecipitation").alias('Avg_MonthPrecipitation'),
    F.avg('HourlyVisibility').alias('Avg_MonthVisibility'),
    F.avg('HourlyWindSpeed').alias('Avg_MonthWindSpeed')
)
display(df_weather)
display(weather_group_day)
display(weather_group_week)
display(weather_group_month)

weather_group_day = weather_group_day.toPandas()

weather_group_day['DATE'] = pd.to_datetime(weather_group_day['date_only'], format = '%Y-%m-%d')
#grouped_day_avg_delay_pd = grouped_day_avg_delay_pd.set_index('FL_DATE')
weather_group_day.sort_values(by=['DATE'], inplace = True)

number_of_weather_stations = df_weather.groupBy('STATION').count()

# COMMAND ----------

plt.plot(weather_group_day['DATE'], weather_group_day['Avg_DailyVisibility'])
plt.xticks(weather_group_day['DATE'][::30])
plt.title('Avg Daily Visibility')
plt.show()

plt.plot(weather_group_day['DATE'], weather_group_day['Avg_DailyWindSpeed'])
plt.xticks(weather_group_day['DATE'][::30])
plt.title('Avg Daily Windspeed')
plt.show()

plt.plot(weather_group_day['DATE'], weather_group_day['Avg_DailyPrecipitation'])
plt.xticks(weather_group_day['DATE'][::30])
plt.title('Avg Daily Rainfall (inchs)')
plt.show()


# COMMAND ----------

##average delay by day
grouped_day_avg_delay_pd['FL_DATE'] = pd.to_datetime(grouped_day_avg_delay_pd['FL_DATE'], format = '%Y-%m-%d')
#grouped_day_avg_delay_pd = grouped_day_avg_delay_pd.set_index('FL_DATE')
grouped_day_avg_delay_pd.sort_values(by=['FL_DATE'], inplace = True)


plt.plot(grouped_day_avg_delay_pd['FL_DATE'], grouped_day_avg_delay_pd['Avg_DEP_DELAY'])
plt.xticks(grouped_day_avg_delay_pd['FL_DATE'][::30])
plt.show()

# COMMAND ----------

import numpy as np

graph_data = grouped_month_avg_delay.toPandas()
graph_data.sort_values(by = ['MONTH'], inplace = True)
graph_dict = graph_data[['Avg_WEATHER_DELAY', 'Avg_CARRIER_DELAY', 'Avg_NAS_DELAY', 'Avg_LATE_AIRCRAFT_DELAY']].to_dict('list')
graph_dict

months = np.arange(len(graph_data['MONTH']))  # the label locations
width = 0.20  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize = (20,10))

for attribute, measurement in graph_dict.items():
    offset = width * multiplier
    rects = ax.bar(months + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=5)
    multiplier += 1

ax.set_xticks(x + width, months)
fig.
ax.legend(loc='best', ncols=1)
plt.show()


# COMMAND ----------

#sort and take top 50 avg dep and arrival delay from both origin and dest locations 
top_50_origin_dep_delay = grouped_origin_avg_delay.sort(F.desc('Avg_DEP_DELAY')).limit(50).toPandas()
top_50_origin_arr_delay = grouped_origin_avg_delay.sort(F.desc('Avg_ARR_DELAY')).limit(50).toPandas()
top_50_dest_dep_delay = grouped_dest_avg_delay.sort(F.desc('Avg_DEP_DELAY')).limit(50).toPandas()
top_50_dest_arr_delay = grouped_dest_avg_delay.sort(F.desc('Avg_ARR_DELAY')).limit(50).toPandas()


# is flight distance related to arrival delay?
sample = df_flights.select(['ARR_DELAY', 'DISTANCE']).sample(.0010).toPandas()
plt.scatter(sample['DISTANCE'], sample['ARR_DELAY'])
plt.title('Distance vs. Arrival Delay')
plt.xlabel('Distance')
plt.ylabel('Arrival Delay')
plt.show()








# COMMAND ----------

delayed = df_otpw.filter(df_otpw.DEP_DEL15 == 0).groupBy('HourlyPresentWeatherType').count()
not_delayed = df_otpw.filter(df_otpw.DEP_DEL15 == 1).groupBy('HourlyPresentWeatherType').count()

delayed_count = df_otpw.filter(df_otpw.DEP_DEL15 == 0).count()
not_delayed_count = df_otpw.filter(df_otpw.DEP_DEL15 == 1).count()

delayed = delayed.sort('count', ascending = False).toPandas()
not_delayed = not_delayed.sort('count', ascending = False).toPandas()


delayed_grouped_weather = df_otpw.groupBy('DEP_DEL15').agg(
    F.avg("HourlyPrecipitation").alias('Avg_HourlyPrecipitation'),
    F.avg('HourlyVisibility').alias('Avg_HourlyVisibility'),
    F.avg('HourlyWindSpeed').alias('Avg_HourlyWindSpeed')
)
delayed['count'] = delayed['count']/delayed_count
not_delayed['count'] = not_delayed['count']/not_delayed_count

display(delayed)

# COMMAND ----------

import re
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

def preprocessing(string):
    if string:
        string = re.sub('[\|a-z0-9\:]', '', string)
        string = string.split(' ')
        string = [x for x in string if x]
    return string

preprocessing_udf = udf(preprocessing, ArrayType(StringType()))


df_otpw = df_otpw.withColumn("Weather_Codes", preprocessing_udf(F.col('HourlyPresentWeatherType')))
df_otpw = df_otpw.withColumn("Sky_Conditions_Codes", preprocessing_udf(F.col('HourlySkyConditions')))

display(df_otpw)

# COMMAND ----------



# COMMAND ----------


na_counts = df_otpw.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) / 1401363 for c in df_otpw.columns])

display(na_counts)


# COMMAND ----------



# COMMAND ----------

delayed = df_otpw.filter(df_otpw.DEP_DEL15 == 0)
not_delayed = df_otpw.filter(df_otpw.DEP_DEL15 == 1)


delayed_column = delayed.select('HourlyDewPointTemperature')
not_delayed_column = not_delayed.select('HourlyDewPointTemperature')

delayed_column = delayed_column.withColumn("HourlyDewPointTemperatureDouble", F.col("HourlyDewPointTemperature").cast("double"))
not_delayed_column = not_delayed_column.withColumn("HourlyDewPointTemperatureDouble", F.col("HourlyDewPointTemperature").cast("double"))

delayed_column = delayed_column.toPandas()
not_delayed_column = not_delayed_column.toPandas()


delayed_column.hist()
not_delayed_column.hist()

# COMMAND ----------

   
import numpy as np
from scipy.stats import ks_2samp

ks_statistic, p_value = ks_2samp(delayed_column['HourlyDewPointTemperatureDouble'], not_delayed_column['HourlyDewPointTemperatureDouble'])

print(f"Kolmogorov–Smirnov Statistic: {ks_statistic}")
print(f"P-value: {p_value}")



# COMMAND ----------

columns = ['HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyPressureChange', 'HourlyPrecipitation', 
'HourlyVisibility', 'HourlyRelativeHumidity', 'HourlyWindSpeed','DailyAverageWindSpeed','DailyAverageDewPointTemperature','DailyDepartureFromNormalAverageTemperature', 'DailyPrecipitation']
#columns = df_weather.schema.names

for column in columns:
    delayed_column = delayed.select(column)
    not_delayed_column = not_delayed.select(column)

    delayed_column = delayed_column.withColumn(column + 'Double', F.col(column).cast("double"))
    not_delayed_column = not_delayed_column.withColumn(column + 'Double', F.col(column).cast("double"))

    delayed_column = delayed_column.toPandas()
    not_delayed_column = not_delayed_column.toPandas()

    delayed_column.hist()
    not_delayed_column.hist()

    ks_statistic, p_value = ks_2samp(delayed_column[column + 'Double'], not_delayed_column[column + 'Double'])

    print(f"Kolmogorov–Smirnov Statistic: {ks_statistic}" + 'for' + column + 'Double')
    print(f"P-value: {p_value}")




# COMMAND ----------



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

