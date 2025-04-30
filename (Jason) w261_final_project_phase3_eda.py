# Databricks notebook source
from pyspark.sql.functions import col
print("Welcome to the W261 final project!") 

# COMMAND ----------

spark.conf.set('spark.sql.caseSensitive', True)

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

# MAGIC %md
# MAGIC
# MAGIC # Know your mount
# MAGIC Here is the mounting for this class, your source for the original data! Remember, you only have Read access, not Write! Also, become familiar with `dbutils` the equivalent of `gcp` in DataProc

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

display(dbutils.fs.ls(f"dbfs:/mnt/mids-w261/OTPW_12M"))

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
# display(df_flights)

# COMMAND ----------

display(dbutils.fs.ls(f"dbfs:/mnt/mids-w261/datasets_final_project_2022"))

# COMMAND ----------

# Weather data
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
# display(df_weather)

# COMMAND ----------

# Stations data      
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
# display(df_stations)

# COMMAND ----------

# OTPW 1 Year
df_otpw_1y = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M_2015.csv.gz")
# display(df_otpw_1y)

# COMMAND ----------

# OTPW 3 Month
df_otpw_3m = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
# display(df_otpw_3m)

# COMMAND ----------

display(dbutils.fs.ls(f"dbfs:/mnt/mids-w261/OTPW_60M"))

# COMMAND ----------

# OPTW 60 Month
# Get separate tables for each variable
df_otpw_60m_part1 = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_60M/part-00000-tid-6348072830472745186-30cd655d-a41e-4b9e-8f85-ca9cebbfe25a-19-1-c000.csv.gz")
df_otpw_60m_part2 = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_60M/part-00001-tid-6348072830472745186-30cd655d-a41e-4b9e-8f85-ca9cebbfe25a-18-1-c000.csv.gz")
df_otpw_60m_part3 = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_60M/part-00002-tid-6348072830472745186-30cd655d-a41e-4b9e-8f85-ca9cebbfe25a-21-1-c000.csv.gz")
df_otpw_60m_part4 = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_60M/part-00003-tid-6348072830472745186-30cd655d-a41e-4b9e-8f85-ca9cebbfe25a-20-1-c000.csv.gz")
df_otpw_60m_part5 = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_60M/part-00004-tid-6348072830472745186-30cd655d-a41e-4b9e-8f85-ca9cebbfe25a-22-1-c000.csv.gz")
# display(df_otpw_60m_part1)


# COMMAND ----------

## Combine into one df
from functools import reduce
from pyspark.sql import DataFrame
dfs = [df_otpw_60m_part1,df_otpw_60m_part2,df_otpw_60m_part3, df_otpw_60m_part4, df_otpw_60m_part5]
df_otpw_60m = reduce(DataFrame.unionAll, dfs)
# display(df_otpw_60m)

# COMMAND ----------

otpw_columns = df_otpw_1y.columns
otpw_columns

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, when, concat

# COMMAND ----------

df_1y = df_otpw_1y.filter(df_otpw_1y.DEP_DELAY_NEW.isNotNull())
df_3m = df_otpw_3m.filter(df_otpw_3m.DEP_DELAY_NEW.isNotNull())
df_60m = df_otpw_60m.filter(df_otpw_60m.DEP_DELAY_NEW.isNotNull())

# COMMAND ----------

## Create Time based variable
import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, when, concat, lag, substring
from pyspark.sql.window import Window

### Create variable to see if tail num is in the air 2 hrs before departure time
windowSpec = Window.partitionBy("TAIL_NUM").orderBy("YEAR", "MONTH", "DAY_OF_MONTH", "DEP_TIME")

# Define a lag window function to look back 2 hours
lag_window_DoM = lag(col("DAY_OF_MONTH"), 1).over(windowSpec)
lag_window_dep = lag(col("DEP_TIME"), 1).over(windowSpec)
lag_window_arr = lag(col("ARR_TIME"), 1).over(windowSpec)
lag_window_delay = lag(col("DEP_DEL15"), 1).over(windowSpec)

# Check if plane is in the air & if it is delayed
df_stg = df_3m.withColumn(
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
df_3m = df_stg.withColumn(
    'is_prev_flight_delayed',
    when((lag_window_delay.cast("double") == 1)
               &
      (lag_window_DoM == F.col("DAY_OF_MONTH")), 1).otherwise(0)

)

# Show the resulting DataFrame
# display(df_3m)


# COMMAND ----------

## Create Time based variable (1 year)


### Create variable to see if tail num is in the air 2 hrs before departure time
windowSpec = Window.partitionBy("TAIL_NUM").orderBy("YEAR", "MONTH", "DAY_OF_MONTH", "DEP_TIME")

# Define a lag window function to look back 2 hours
lag_window_DoM = lag(col("DAY_OF_MONTH"), 1).over(windowSpec)
lag_window_dep = lag(col("DEP_TIME"), 1).over(windowSpec)
lag_window_arr = lag(col("ARR_TIME"), 1).over(windowSpec)
lag_window_delay = lag(col("DEP_DEL15"), 1).over(windowSpec)

# Check if plane is in the air & if it is delayed
df_stg = df_1y.withColumn(
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
df_1y = df_stg.withColumn(
    'is_prev_flight_delayed',
    when((lag_window_delay.cast("double") == 1)
               &
      (lag_window_DoM == F.col("DAY_OF_MONTH")), 1).otherwise(0)

)

# Show the resulting DataFrame
# display(df_1y)


# COMMAND ----------

## Create Time based variable (60 month)


### Create variable to see if tail num is in the air 2 hrs before departure time
windowSpec = Window.partitionBy("TAIL_NUM").orderBy("YEAR", "MONTH", "DAY_OF_MONTH", "DEP_TIME")

# Define a lag window function to look back 2 hours
lag_window_DoM = lag(col("DAY_OF_MONTH"), 1).over(windowSpec)
lag_window_dep = lag(col("DEP_TIME"), 1).over(windowSpec)
lag_window_arr = lag(col("ARR_TIME"), 1).over(windowSpec)
lag_window_delay = lag(col("DEP_DEL15"), 1).over(windowSpec)

# Check if plane is in the air & if it is delayed
df_stg = df_60m.withColumn(
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
df_60m = df_stg.withColumn(
    'is_prev_flight_delayed',
    when((lag_window_delay.cast("double") == 1)
               &
      (lag_window_DoM == F.col("DAY_OF_MONTH")), 1).otherwise(0)

)

# Show the resulting DataFrame
# display(df_60m)


# COMMAND ----------

#Drop Daily/Montly Features along with features with unusuable data for 3 month data
df_3m = df_3m.drop('Sunrise',
 'Sunset',
 'DailyAverageDewPointTemperature',
 'DailyAverageDryBulbTemperature',
 'DailyAverageRelativeHumidity',
 'DailyAverageSeaLevelPressure',
 'DailyAverageStationPressure',
 'DailyAverageWetBulbTemperature',
 'DailyAverageWindSpeed',
 'DailyCoolingDegreeDays',
 'DailyDepartureFromNormalAverageTemperature',
 'DailyHeatingDegreeDays',
 'DailyMaximumDryBulbTemperature',
 'DailyMinimumDryBulbTemperature',
 'DailyPeakWindDirection',
 'DailyPeakWindSpeed',
 'DailyPrecipitation',
 'DailySnowDepth',
 'DailySnowfall',
 'DailySustainedWindDirection',
 'DailySustainedWindSpeed',
 'DailyWeather',
 'MonthlyAverageRH',
 'MonthlyDaysWithGT001Precip',
 'MonthlyDaysWithGT010Precip',
 'MonthlyDaysWithGT32Temp',
 'MonthlyDaysWithGT90Temp',
 'MonthlyDaysWithLT0Temp',
 'MonthlyDaysWithLT32Temp',
 'MonthlyDepartureFromNormalAverageTemperature',
 'MonthlyDepartureFromNormalCoolingDegreeDays',
 'MonthlyDepartureFromNormalHeatingDegreeDays',
 'MonthlyDepartureFromNormalMaximumTemperature',
 'MonthlyDepartureFromNormalMinimumTemperature',
 'MonthlyDepartureFromNormalPrecipitation',
 'MonthlyDewpointTemperature',
 'MonthlyGreatestPrecip',
 'MonthlyGreatestPrecipDate',
 'MonthlyGreatestSnowDepth',
 'MonthlyGreatestSnowDepthDate',
 'MonthlyGreatestSnowfall',
 'MonthlyGreatestSnowfallDate',
 'MonthlyMaxSeaLevelPressureValue',
 'MonthlyMaxSeaLevelPressureValueDate',
 'MonthlyMaxSeaLevelPressureValueTime',
 'MonthlyMaximumTemperature',
 'MonthlyMeanTemperature',
 'MonthlyMinSeaLevelPressureValue',
 'MonthlyMinSeaLevelPressureValueDate',
 'MonthlyMinSeaLevelPressureValueTime',
 'MonthlyMinimumTemperature',
 'MonthlySeaLevelPressure',
 'MonthlyStationPressure',
 'MonthlyTotalLiquidPrecipitation',
 'MonthlyTotalSnowfall',
 'MonthlyWetBulb',
 'AWND',
 'CDSD',
 'CLDD',
 'DSNW',
 'HDSD',
 'HTDD',
 'NormalsCoolingDegreeDay',
 'NormalsHeatingDegreeDay',
 'ShortDurationEndDate005',
 'ShortDurationEndDate010',
 'ShortDurationEndDate015',
 'ShortDurationEndDate020',
 'ShortDurationEndDate030',
 'ShortDurationEndDate045',
 'ShortDurationEndDate060',
 'ShortDurationEndDate080',
 'ShortDurationEndDate100',
 'ShortDurationEndDate120',
 'ShortDurationEndDate150',
 'ShortDurationEndDate180',
 'ShortDurationPrecipitationValue005',
 'ShortDurationPrecipitationValue010',
 'ShortDurationPrecipitationValue015',
 'ShortDurationPrecipitationValue020',
 'ShortDurationPrecipitationValue030',
 'ShortDurationPrecipitationValue045',
 'ShortDurationPrecipitationValue060',
 'ShortDurationPrecipitationValue080',
 'ShortDurationPrecipitationValue100',
 'ShortDurationPrecipitationValue120',
 'ShortDurationPrecipitationValue150',
 'ShortDurationPrecipitationValue180',
 'REM',
 'BackupDirection',
 'BackupDistance',
 'BackupDistanceUnit',
 'BackupElements',
 'BackupElevation',
 'BackupEquipment',
 'BackupLatitude',
 'BackupLongitude',
 'BackupName',
 'WindEquipmentChangeDate',
 '_row_desc')

# COMMAND ----------

#Drop Daily/Montly Features along with features with unusuable data for 1 year data
df_1y = df_1y.drop('Sunrise',
 'Sunset',
 'DailyAverageDewPointTemperature',
 'DailyAverageDryBulbTemperature',
 'DailyAverageRelativeHumidity',
 'DailyAverageSeaLevelPressure',
 'DailyAverageStationPressure',
 'DailyAverageWetBulbTemperature',
 'DailyAverageWindSpeed',
 'DailyCoolingDegreeDays',
 'DailyDepartureFromNormalAverageTemperature',
 'DailyHeatingDegreeDays',
 'DailyMaximumDryBulbTemperature',
 'DailyMinimumDryBulbTemperature',
 'DailyPeakWindDirection',
 'DailyPeakWindSpeed',
 'DailyPrecipitation',
 'DailySnowDepth',
 'DailySnowfall',
 'DailySustainedWindDirection',
 'DailySustainedWindSpeed',
 'DailyWeather',
 'MonthlyAverageRH',
 'MonthlyDaysWithGT001Precip',
 'MonthlyDaysWithGT010Precip',
 'MonthlyDaysWithGT32Temp',
 'MonthlyDaysWithGT90Temp',
 'MonthlyDaysWithLT0Temp',
 'MonthlyDaysWithLT32Temp',
 'MonthlyDepartureFromNormalAverageTemperature',
 'MonthlyDepartureFromNormalCoolingDegreeDays',
 'MonthlyDepartureFromNormalHeatingDegreeDays',
 'MonthlyDepartureFromNormalMaximumTemperature',
 'MonthlyDepartureFromNormalMinimumTemperature',
 'MonthlyDepartureFromNormalPrecipitation',
 'MonthlyDewpointTemperature',
 'MonthlyGreatestPrecip',
 'MonthlyGreatestPrecipDate',
 'MonthlyGreatestSnowDepth',
 'MonthlyGreatestSnowDepthDate',
 'MonthlyGreatestSnowfall',
 'MonthlyGreatestSnowfallDate',
 'MonthlyMaxSeaLevelPressureValue',
 'MonthlyMaxSeaLevelPressureValueDate',
 'MonthlyMaxSeaLevelPressureValueTime',
 'MonthlyMaximumTemperature',
 'MonthlyMeanTemperature',
 'MonthlyMinSeaLevelPressureValue',
 'MonthlyMinSeaLevelPressureValueDate',
 'MonthlyMinSeaLevelPressureValueTime',
 'MonthlyMinimumTemperature',
 'MonthlySeaLevelPressure',
 'MonthlyStationPressure',
 'MonthlyTotalLiquidPrecipitation',
 'MonthlyTotalSnowfall',
 'MonthlyWetBulb',
 'AWND',
 'CDSD',
 'CLDD',
 'DSNW',
 'HDSD',
 'HTDD',
 'NormalsCoolingDegreeDay',
 'NormalsHeatingDegreeDay',
 'ShortDurationEndDate005',
 'ShortDurationEndDate010',
 'ShortDurationEndDate015',
 'ShortDurationEndDate020',
 'ShortDurationEndDate030',
 'ShortDurationEndDate045',
 'ShortDurationEndDate060',
 'ShortDurationEndDate080',
 'ShortDurationEndDate100',
 'ShortDurationEndDate120',
 'ShortDurationEndDate150',
 'ShortDurationEndDate180',
 'ShortDurationPrecipitationValue005',
 'ShortDurationPrecipitationValue010',
 'ShortDurationPrecipitationValue015',
 'ShortDurationPrecipitationValue020',
 'ShortDurationPrecipitationValue030',
 'ShortDurationPrecipitationValue045',
 'ShortDurationPrecipitationValue060',
 'ShortDurationPrecipitationValue080',
 'ShortDurationPrecipitationValue100',
 'ShortDurationPrecipitationValue120',
 'ShortDurationPrecipitationValue150',
 'ShortDurationPrecipitationValue180',
 'REM',
 'BackupDirection',
 'BackupDistance',
 'BackupDistanceUnit',
 'BackupElements',
 'BackupElevation',
 'BackupEquipment',
 'BackupLatitude',
 'BackupLongitude',
 'BackupName',
 'WindEquipmentChangeDate',
 '_row_desc')

# COMMAND ----------

#Drop Daily/Montly Features along with features with unusuable data for 60 month data
df_60m = df_60m.drop('Sunrise',
 'Sunset',
 'DailyAverageDewPointTemperature',
 'DailyAverageDryBulbTemperature',
 'DailyAverageRelativeHumidity',
 'DailyAverageSeaLevelPressure',
 'DailyAverageStationPressure',
 'DailyAverageWetBulbTemperature',
 'DailyAverageWindSpeed',
 'DailyCoolingDegreeDays',
 'DailyDepartureFromNormalAverageTemperature',
 'DailyHeatingDegreeDays',
 'DailyMaximumDryBulbTemperature',
 'DailyMinimumDryBulbTemperature',
 'DailyPeakWindDirection',
 'DailyPeakWindSpeed',
 'DailyPrecipitation',
 'DailySnowDepth',
 'DailySnowfall',
 'DailySustainedWindDirection',
 'DailySustainedWindSpeed',
 'DailyWeather',
 'MonthlyAverageRH',
 'MonthlyDaysWithGT001Precip',
 'MonthlyDaysWithGT010Precip',
 'MonthlyDaysWithGT32Temp',
 'MonthlyDaysWithGT90Temp',
 'MonthlyDaysWithLT0Temp',
 'MonthlyDaysWithLT32Temp',
 'MonthlyDepartureFromNormalAverageTemperature',
 'MonthlyDepartureFromNormalCoolingDegreeDays',
 'MonthlyDepartureFromNormalHeatingDegreeDays',
 'MonthlyDepartureFromNormalMaximumTemperature',
 'MonthlyDepartureFromNormalMinimumTemperature',
 'MonthlyDepartureFromNormalPrecipitation',
 'MonthlyDewpointTemperature',
 'MonthlyGreatestPrecip',
 'MonthlyGreatestPrecipDate',
 'MonthlyGreatestSnowDepth',
 'MonthlyGreatestSnowDepthDate',
 'MonthlyGreatestSnowfall',
 'MonthlyGreatestSnowfallDate',
 'MonthlyMaxSeaLevelPressureValue',
 'MonthlyMaxSeaLevelPressureValueDate',
 'MonthlyMaxSeaLevelPressureValueTime',
 'MonthlyMaximumTemperature',
 'MonthlyMeanTemperature',
 'MonthlyMinSeaLevelPressureValue',
 'MonthlyMinSeaLevelPressureValueDate',
 'MonthlyMinSeaLevelPressureValueTime',
 'MonthlyMinimumTemperature',
 'MonthlySeaLevelPressure',
 'MonthlyStationPressure',
 'MonthlyTotalLiquidPrecipitation',
 'MonthlyTotalSnowfall',
 'MonthlyWetBulb',
 'AWND',
 'CDSD',
 'CLDD',
 'DSNW',
 'HDSD',
 'HTDD',
 'NormalsCoolingDegreeDay',
 'NormalsHeatingDegreeDay',
 'ShortDurationEndDate005',
 'ShortDurationEndDate010',
 'ShortDurationEndDate015',
 'ShortDurationEndDate020',
 'ShortDurationEndDate030',
 'ShortDurationEndDate045',
 'ShortDurationEndDate060',
 'ShortDurationEndDate080',
 'ShortDurationEndDate100',
 'ShortDurationEndDate120',
 'ShortDurationEndDate150',
 'ShortDurationEndDate180',
 'ShortDurationPrecipitationValue005',
 'ShortDurationPrecipitationValue010',
 'ShortDurationPrecipitationValue015',
 'ShortDurationPrecipitationValue020',
 'ShortDurationPrecipitationValue030',
 'ShortDurationPrecipitationValue045',
 'ShortDurationPrecipitationValue060',
 'ShortDurationPrecipitationValue080',
 'ShortDurationPrecipitationValue100',
 'ShortDurationPrecipitationValue120',
 'ShortDurationPrecipitationValue150',
 'ShortDurationPrecipitationValue180',
 'REM',
 'BackupDirection',
 'BackupDistance',
 'BackupDistanceUnit',
 'BackupElements',
 'BackupElevation',
 'BackupEquipment',
 'BackupLatitude',
 'BackupLongitude',
 'BackupName',
 'WindEquipmentChangeDate',
 '_row_desc')

# COMMAND ----------

#Drop arrival related features
df_1y= df_1y.drop('TAXI_IN',
'CRS_ARR_TIME',
'ARR_TIME',
'ARR_DELAY',
'ARR_DELAY_NEW',
'ARR_DEL15',
'ARR_DELAY_GROUP',
'CRS_ELAPSED_TIME',
'ACTUAL_ELAPSED_TIME',
'AIR_TIME')

df_3m = df_3m.drop('TAXI_IN',
'CRS_ARR_TIME',
'ARR_TIME',
'ARR_DELAY',
'ARR_DELAY_NEW',
'ARR_DEL15',
'ARR_DELAY_GROUP',
'CRS_ELAPSED_TIME',
'ACTUAL_ELAPSED_TIME',
'AIR_TIME')

df_60m = df_60m.drop('TAXI_IN',
'CRS_ARR_TIME',
'ARR_TIME',
'ARR_DELAY',
'ARR_DELAY_NEW',
'ARR_DEL15',
'ARR_DELAY_GROUP',
'CRS_ELAPSED_TIME',
'ACTUAL_ELAPSED_TIME',
'AIR_TIME')

# COMMAND ----------

df_3m = df_3m.drop('FL_DATE', 'OP_CARRIER', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_NM', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'CANCELLATION_CODE')

df_1y = df_1y.drop('FL_DATE', 'OP_CARRIER', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_NM', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'CANCELLATION_CODE')

df_60m = df_60m.drop('FL_DATE', 'OP_CARRIER', 'TAIL_NUM', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_NM', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'CANCELLATION_CODE')

# COMMAND ----------

df_3m = df_3m.drop('origin_airport_name',
'origin_station_name',
'origin_station_id',
'origin_iata_code',
'origin_icao',
'origin_type',
'origin_region',
'origin_station_lat',
'origin_station_lon',
'origin_airport_lat',
'origin_airport_lon',
'origin_station_dis',
'dest_airport_name',
'dest_station_name',
'dest_station_id',
'dest_iata_code',
'dest_icao',
'dest_type',
'dest_region',
'dest_station_lat',
'dest_station_lon',
'dest_airport_lat',
'dest_airport_lon',
'dest_station_dis')

df_1y = df_1y.drop('origin_airport_name',
'origin_station_name',
'origin_station_id',
'origin_iata_code',
'origin_icao',
'origin_type',
'origin_region',
'origin_station_lat',
'origin_station_lon',
'origin_airport_lat',
'origin_airport_lon',
'origin_station_dis',
'dest_airport_name',
'dest_station_name',
'dest_station_id',
'dest_iata_code',
'dest_icao',
'dest_type',
'dest_region',
'dest_station_lat',
'dest_station_lon',
'dest_airport_lat',
'dest_airport_lon',
'dest_station_dis')

df_60m = df_60m.drop('origin_airport_name',
'origin_station_name',
'origin_station_id',
'origin_iata_code',
'origin_icao',
'origin_type',
'origin_region',
'origin_station_lat',
'origin_station_lon',
'origin_airport_lat',
'origin_airport_lon',
'origin_station_dis',
'dest_airport_name',
'dest_station_name',
'dest_station_id',
'dest_iata_code',
'dest_icao',
'dest_type',
'dest_region',
'dest_station_lat',
'dest_station_lon',
'dest_airport_lat',
'dest_airport_lon',
'dest_station_dis')


# COMMAND ----------

df_3m = df_3m.drop("Station","Date","Name","Source","Report_Type")

df_1y = df_1y.drop("Station","Date","Name","Source","Report_Type")

df_60m = df_60m.drop("Station","Date","Name","Source","Report_Type")


# COMMAND ----------

df_3m = df_3m.drop("HourlyPressureChange","HourlyPressureTendency")

df_1y = df_1y.drop("HourlyPressureChange","HourlyPressureTendency")

df_60m = df_60m.drop("HourlyPressureChange","HourlyPressureTendency")

# COMMAND ----------

#Convert hourly columns to doubles
hourly_columns = [
    'HourlyAltimeterSetting',
    'HourlyDewPointTemperature',
    'HourlyDryBulbTemperature',
    'HourlyPrecipitation',
    'HourlyRelativeHumidity',
    'HourlySeaLevelPressure',
    'HourlyStationPressure',
    'HourlyVisibility',
    'HourlyWetBulbTemperature',
    'HourlyWindDirection',
    'HourlyWindGustSpeed',
    'HourlyWindSpeed'
]

for column in hourly_columns:
    df_3m = df_3m.withColumn(column, col(column).cast('double'))
    df_1y = df_1y.withColumn(column, col(column).cast('double'))
    df_60m = df_60m.withColumn(column, col(column).cast('double'))

# COMMAND ----------

df_3m = df_3m.drop('Longitude','Latitude','Elevation')
df_1y = df_1y.drop('Longitude','Latitude','Elevation')
df_60m = df_60m.drop('Longitude','Latitude','Elevation')

# COMMAND ----------

df_3m = df_3m.drop('ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW', 'LONGEST_ADD_GTIM',
'sched_depart_date_time',
'sched_depart_date_time_UTC',
'four_hours_prior_depart_UTC',
'two_hours_prior_depart_UTC', 'TAXI_OUT',
'WHEELS_OFF',
'WHEELS_ON',
'CANCELLED',
'DIVERTED')

df_1y = df_1y.drop('ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW', 'LONGEST_ADD_GTIME',
'sched_depart_date_time',
'sched_depart_date_time_UTC',
'four_hours_prior_depart_UTC',
'two_hours_prior_depart_UTC', 'TAXI_OUT',
'WHEELS_OFF',
'WHEELS_ON',
'CANCELLED',
'DIVERTED')

df_60m = df_60m.drop('ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW', 'LONGEST_ADD_GTIME',
'sched_depart_date_time',
'sched_depart_date_time_UTC',
'four_hours_prior_depart_UTC',
'two_hours_prior_depart_UTC', 'TAXI_OUT',
'WHEELS_OFF',
'WHEELS_ON',
'CANCELLED',
'DIVERTED')

# COMMAND ----------

##cleaning weather and sky conditions codes 
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


df_3m = df_3m.withColumn("Weather_Codes", preprocessing_udf(F.col('HourlyPresentWeatherType')))
df_3m = df_3m.withColumn("Sky_Conditions_Codes", preprocessing_udf(F.col('HourlySkyConditions')))

df_1y = df_1y.withColumn("Weather_Codes", preprocessing_udf(F.col('HourlyPresentWeatherType')))
df_1y = df_1y.withColumn("Sky_Conditions_Codes", preprocessing_udf(F.col('HourlySkyConditions')))

df_60m = df_60m.withColumn("Weather_Codes", preprocessing_udf(F.col('HourlyPresentWeatherType')))
df_60m = df_60m.withColumn("Sky_Conditions_Codes", preprocessing_udf(F.col('HourlySkyConditions')))

# display(df_1y)

# COMMAND ----------

# Drop raw weather type and sky condition columns
df_3m = df_3m.drop('HourlySkyConditions','HourlyPresentWeatherType')
df_1y = df_1y.drop('HourlySkyConditions','HourlyPresentWeatherType')
df_60m = df_60m.drop('HourlySkyConditions','HourlyPresentWeatherType')

# COMMAND ----------

#Drop future related features and redundant IDs
df_3m = df_3m.drop(
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
'LONGEST_ADD_GTIME',
'ORIGIN_CITY_MARKET_ID',
'ORIGIN_STATE_FIPS',
'ORIGIN_WAC',
'DEST_CITY_MARKET_ID',
'DEST_STATE_FIPS',
'DEST_WAC',
'FLIGHTS',
'DEP_DELAY_GROUP')

df_1y = df_1y.drop(
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
'LONGEST_ADD_GTIME',
'ORIGIN_CITY_MARKET_ID',
'ORIGIN_STATE_FIPS',
'ORIGIN_WAC',
'DEST_CITY_MARKET_ID',
'DEST_STATE_FIPS',
'DEST_WAC','FLIGHTS',
'DEP_DELAY_GROUP')

df_60m = df_60m.drop(
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
'LONGEST_ADD_GTIME',
'ORIGIN_CITY_MARKET_ID',
'ORIGIN_STATE_FIPS',
'ORIGIN_WAC',
'DEST_CITY_MARKET_ID',
'DEST_STATE_FIPS',
'DEST_WAC','FLIGHTS',
'DEP_DELAY_GROUP')

# COMMAND ----------

from pyspark.sql.functions import array

df_3m = df_3m.withColumn("Sky_Conditions_Codes", F.coalesce(col("Sky_Conditions_Codes"), array()))
df_3m = df_3m.withColumn("Weather_Codes", F.coalesce(col("Weather_Codes"), array()))

# COMMAND ----------

skycodes = [
    x[0] for x in 
    df_3m.select(F.explode("Sky_Conditions_Codes").alias("Sky_Condition_Codes")).distinct().orderBy("Sky_Condition_Codes").collect()]

# COMMAND ----------

weather_codes = [
    x[0] for x in 
    df_3m.select(F.explode("Weather_Codes").alias("Weather_Codes")).distinct().orderBy("Weather_codes").collect()]

# COMMAND ----------

display(df_sep_60m)

# COMMAND ----------

df_sep_3m = df_3m.select("*", *[
    F.array_contains("Weather_Codes", weathercode).alias("WeatherCode_{}".format(weathercode)).cast("integer") 
    for weathercode in weather_codes],*[
    F.array_contains("Sky_Conditions_Codes", skycode).alias("SkyCode_{}".format(skycode)).cast("integer") 
    for skycode in skycodes]
)
# display(df_sep_3m)

# COMMAND ----------

df_1y = df_1y.withColumn("Sky_Conditions_Codes", F.coalesce(col("Sky_Conditions_Codes"), array()))
df_1y = df_1y.withColumn("Weather_Codes", F.coalesce(col("Weather_Codes"), array()))

# COMMAND ----------

skycodes1y = [
    x[0] for x in 
    df_1y.select(F.explode("Sky_Conditions_Codes").alias("Sky_Condition_Codes")).distinct().orderBy("Sky_Condition_Codes").collect()]

weather_codes1y = [
    x[0] for x in 
    df_1y.select(F.explode("Weather_Codes").alias("Weather_Codes")).distinct().orderBy("Weather_codes").collect()]

# COMMAND ----------

df_sep_1y = df_1y.select("*", *[
    F.array_contains("Weather_Codes", weathercode).alias("WeatherCode_{}".format(weathercode)).cast("integer") 
    for weathercode in weather_codes1y],*[
    F.array_contains("Sky_Conditions_Codes", skycode).alias("SkyCode_{}".format(skycode)).cast("integer") 
    for skycode in skycodes1y]
)
# display(df_sep_1y)

# COMMAND ----------

df_60m = df_60m.withColumn("Sky_Conditions_Codes", F.coalesce(col("Sky_Conditions_Codes"), array()))
df_60m = df_60m.withColumn("Weather_Codes", F.coalesce(col("Weather_Codes"), array()))

# COMMAND ----------

skycodes60m = [
    x[0] for x in 
    df_60m.select(F.explode("Sky_Conditions_Codes").alias("Sky_Condition_Codes")).distinct().orderBy("Sky_Condition_Codes").collect()]

weather_codes60m = [
    x[0] for x in 
    df_60m.select(F.explode("Weather_Codes").alias("Weather_Codes")).distinct().orderBy("Weather_codes").collect()]

# COMMAND ----------

df_sep_60m = df_60m.select("*", *[
    F.array_contains("Weather_Codes", weathercode).alias("WeatherCode_{}".format(weathercode)).cast("integer") 
    for weathercode in weather_codes1y],*[
    F.array_contains("Sky_Conditions_Codes", skycode).alias("SkyCode_{}".format(skycode)).cast("integer") 
    for skycode in skycodes1y]
)
# display(df_sep_60m)

# COMMAND ----------

#Drop original weather and sky code columns
df_sep_3m = df_sep_3m.drop('Weather_Codes','Sky_Conditions_Codes')
df_sep_1y = df_sep_1y.drop('Weather_Codes','Sky_Conditions_Codes')
df_sep_60m = df_sep_60m.drop('Weather_Codes','Sky_Conditions_Codes')

# COMMAND ----------

convert_columns = [
    'QUARTER',
 'DAY_OF_MONTH',
 'DAY_OF_WEEK',
 'CRS_DEP_TIME',
 'DEP_DEL15',
 'DISTANCE',
 'DISTANCE_GROUP',
 'YEAR',
 'MONTH'
]

for column in convert_columns:
    df_sep_3m = df_sep_3m.withColumn(column, col(column).cast('double'))
    df_sep_1y = df_sep_1y.withColumn(column, col(column).cast('double'))
    df_sep_60m = df_sep_60m.withColumn(column, col(column).cast('double'))

# COMMAND ----------

df_sep_1y = df_sep_1y.na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

df_sep_3m = df_sep_3m.na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

df_sep_60m = df_sep_60m.na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

# Read the CSV files into DataFrames
rankings = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/WSJ_Rankings.csv")
holidays = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/Holidays.csv")

# COMMAND ----------



# Join tables for Rankings
df_sep_3m = df_sep_3m.join(rankings, (df_sep_3m.OP_UNIQUE_CARRIER == rankings.Carrier) & (df_sep_3m.YEAR == rankings.Ranking_YEAR), how='left')
df_sep_1y = df_sep_1y.join(rankings, (df_sep_1y.OP_UNIQUE_CARRIER == rankings.Carrier) & (df_sep_1y.YEAR == rankings.Ranking_YEAR), how='left')
df_sep_60m = df_sep_60m.join(rankings, (df_sep_60m.OP_UNIQUE_CARRIER == rankings.Carrier) & (df_sep_60m.YEAR == rankings.Ranking_YEAR), how='left')

# Handle None Rank values
df_sep_3m = df_sep_3m.fillna({ 'Rank': 10 })
df_sep_60m = df_sep_60m.fillna({ 'Rank': 10 })
df_sep_1y = df_sep_1y.fillna({ 'Rank': 10 })

# Join tables for Holidays
df_sep_3m = df_sep_3m.join(holidays, (df_sep_3m.DAY_OF_MONTH == holidays.Day) & (df_sep_3m.F_MONTH == holidays.Holiday_Month) & (df_sep_3m.YEAR == holidays.Holiday_Year), how='left')
df_sep_1y = df_sep_1y.join(holidays, (df_sep_1y.DAY_OF_MONTH == holidays.Day) & (df_sep_1y.F_MONTH == holidays.Holiday_Month) & (df_sep_1y.YEAR == holidays.Holiday_Year), how='left')
df_sep_60m = df_sep_60m.join(holidays, (df_sep_60m.DAY_OF_MONTH == holidays.Day) & (df_sep_60m.F_MONTH == holidays.Holiday_Month) & (df_sep_60m.YEAR == holidays.Holiday_Year), how='left')

# COMMAND ----------

 

# COMMAND ----------

## Delete redundant columns
df_sep_3m =  df_sep_3m.drop('Full Name', 'Carrier', 'Date', 'WeekDay', 'Day', 'Holiday_Year', 'Holiday_Month', 'Ranking_YEAR')
df_sep_60m = df_sep_60m.drop('Full Name', 'Carrier','Date', 'WeekDay', 'Day', 'Holiday_Year', 'Holiday_Month', 'Ranking_YEAR')
df_sep_1y =  df_sep_1y.drop('Full Name', 'Carrier', 'Date', 'WeekDay', 'Day', 'Holiday_Year', 'Holiday_Month', 'Ranking_YEAR')

# COMMAND ----------

display(df_sep_3m['YEAR'])

# COMMAND ----------

display(df_sep_1y)

# COMMAND ----------

display(df_sep_3m)

# COMMAND ----------

display(df_sep_60m)

# COMMAND ----------

df_sep_1y.write.parquet(f"{team_blob_url}/1year_cleaned_new_1")
df_sep_3m.write.parquet(f"{team_blob_url}/3month_cleaned_new_1")
df_sep_60m.write.parquet(f"{team_blob_url}/60month_cleaned_new_1")

# COMMAND ----------

df_sep_3m_eda_new = spark.read.parquet(f"{team_blob_url}/3month_cleaned_new_1")


# COMMAND ----------

# df_sep_1y_eda_new = spark.read.parquet(f"{team_blob_url}/1year_cleaned_new_1")
df_sep_1y_eda_new = spark.read.parquet(f"{team_blob_url}/new_flight_data_1")


# COMMAND ----------

display(df_sep_3m_eda_new)

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

display(df_otpw_60m_with_ranks)

# COMMAND ----------

##### Delays by Rank
delay_by_Rank = df_otpw_60m_with_ranks.groupby("Rank").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
display(delay_by_Rank)

# COMMAND ----------

import matplotlib.pyplot as plt
## Delay by Day of Week Graph
vals = delay_by_Rank.select(delay_by_Rank[1]).collect()
DoW = delay_by_Rank.select(delay_by_Rank[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
DoW = [d[0] for d in DoW]
plt.bar(DoW, vals, color='green')
plt.xlabel('WSJ Ranking')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_WSJ_Rank")
display(plt.show())

# COMMAND ----------

##### Delays by Holiday
delay_by_Holiday = df_otpw_60m_with_ranks.groupby("Holiday").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
display(delay_by_Holiday)

# COMMAND ----------

delay_by_Holiday = delay_by_Holiday.na.drop(subset=['Holiday'])


# COMMAND ----------

## Delay by Day of Week Graph
vals = delay_by_Holiday.select(delay_by_Holiday[1]).collect()
DoW = delay_by_Holiday.select(delay_by_Holiday[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
DoW = [d[0] for d in DoW]
plt.bar(DoW, vals, color='black')
plt.xlabel('Holiday')
plt.xticks(rotation=90)
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_Holiday")
display(plt.show())

# COMMAND ----------

first_10 = df_sep_1y_eda_new.limit(10)
first_10.toPandas().to_csv('/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/extra_credit.csv')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Resulting Dataframes
# MAGIC
# MAGIC df_3m contains cleaned 3 month data. 
# MAGIC
# MAGIC df_sep_3m contains cleaned 3 month data with dummies for weather/sky codes
# MAGIC
# MAGIC df_1y contains cleaned 1 year data.
# MAGIC
# MAGIC df_sep_1y contains cleaned 1 year data with dummies for weather/sky codes
# MAGIC
# MAGIC df_sep_1y is written to blob storage as "1year_cleaned"
# MAGIC
# MAGIC df_sep_3m is written to blob storage as "3month_cleaned"
# MAGIC
# MAGIC Weather side is cleaned up including categorical variables. Most of flight data is dropped/cleaned based on Jason's notebook. Added time based feature from Jason's notebook. Columns are converted to doubles where appropriate.  

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer, MinMaxScaler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pyspark.sql.functions as F

from hyperopt import fmin, tpe, Trials, SparkTrials, hp
import mlflow
import mlflow.spark

import tensorflow as tf
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential
tf.random.set_seed(42)

import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, when, concat
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

df_sep_3m_eda = spark.read.parquet(f"{team_blob_url}/3month_cleaned")

# cols = df_sep_3m_eda.columns

# print(cols)

# COMMAND ----------

df_sep_3m_eda.display()

# COMMAND ----------

import matplotlib.pyplot as plt
import math

hist_columns = [
#     'QUARTER',
#  'DAY_OF_MONTH',
#  'DAY_OF_WEEK',
 'OP_UNIQUE_CARRIER',
#  'CRS_DEP_TIME',
#  'DEP_DEL15',
#  'DISTANCE',
 'DISTANCE_GROUP',
#  'F_MONTH',
#  'HourlyAltimeterSetting',
 'HourlyDewPointTemperature',
#  'HourlyDryBulbTemperature',
#  'HourlyPrecipitation',
 'HourlyRelativeHumidity',
#  'HourlySeaLevelPressure',
#  'HourlyStationPressure',
#  'HourlyVisibility',
#  'HourlyWetBulbTemperature',
#  'HourlyWindDirection',
#  'HourlyWindGustSpeed',
 'HourlyWindSpeed',
 'pagerank']

def plot_histograms(df):
    # Convert the Spark DataFrame to a Pandas DataFrame
    pandas_df = df.toPandas()
    
    # Get the number of columns in the DataFrame
    num_columns = len(pandas_df.columns)
    
    # Calculate the number of rows and columns for subplots
    num_rows = math.ceil(num_columns / 3)  # Adjust the number of columns per row as needed
    
    # Create subplots with appropriate layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    # Iterate through each column in the DataFrame
    for i, column in enumerate(hist_columns):
        # Plot a histogram for the numeric column
        axes[i].hist(pandas_df[column], bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)
    
    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


# Call the function to plot histograms for every variable
plot_histograms(df_otpw_60m_with_ranks)

# COMMAND ----------

import matplotlib.pyplot as plt
import math

# Define the columns for which you want to plot histograms
hist_columns = ['OP_UNIQUE_CARRIER', 'DISTANCE_GROUP', 'HourlyDewPointTemperature', 
                'HourlyRelativeHumidity', 'HourlyWindSpeed', 'pagerank']

def plot_histograms(df):
    # Convert the Spark DataFrame to a Pandas DataFrame
    pandas_df = df.select(hist_columns).toPandas()  # Limit to selected columns
    
    # Create subplots with appropriate layout
    num_columns = len(hist_columns)
    num_rows = math.ceil(num_columns / 3)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()
    
    # Plot histograms for selected columns
    for i, column in enumerate(hist_columns):
        axes[i].hist(pandas_df[column], bins=10, color='skyblue', edgecolor='black')  # Reduce the number of bins
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)
    
    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Call the function to plot histograms for selected columns
plot_histograms(df_otpw_60m_with_ranks)

# COMMAND ----------

eda_df_3m = df_otpw_60m_with_ranks.select('CRS_DEP_TIME', 'DISTANCE', 'HourlyAltimeterSetting',
'HourlyDewPointTemperature',
'HourlyDryBulbTemperature',
'HourlyPrecipitation',
'HourlyRelativeHumidity',
'HourlySeaLevelPressure',
'HourlyStationPressure',
'HourlyVisibility',
'HourlyWetBulbTemperature',
'HourlyWindDirection',
'HourlyWindGustSpeed',
'HourlyWindSpeed',
'pagerank')

# COMMAND ----------

import matplotlib.pyplot as plt
pd_df = eda_df_3m.toPandas()

# Calculate correlation matrix
plt.figure(figsize=(28, 26))
plt.matshow(pd_df.corr())
plt.show()

# COMMAND ----------

df_sep_3m_eda_new_1 = df_sep_3m_eda_new.drop(
'OP_UNIQUE_CARRIER',
'OP_CARRIER_AIRLINE_ID',
'OP_CARRIER_FL_NUM',
'ORIGIN_AIRPORT_ID',
'ORIGIN',
'DEST_AIRPORT_ID',
'DEST')

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# Create VectorAssembler with handleInvalid='skip'
vector_assembler = VectorAssembler(inputCols=eda_df_3m.columns, outputCol="features", handleInvalid='skip')

# Apply VectorAssembler to the dataset
assembled_df = vector_assembler.transform(eda_df_3m.dropna())

# Calculate correlation matrix
correlation_matrix = Correlation.corr(assembled_df, "features").head()

# Extract the correlation matrix as a dense matrix
corr_matrix = correlation_matrix[0].toArray()

# Get the list of feature names
feature_names = eda_df_3m.columns

# Convert the correlation matrix to a DataFrame with feature names as index and columns
corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)

# Display the correlation matrix DataFrame
print("Correlation Matrix:")
print(corr_df)

# Save the correlation matrix as a CSV
corr_df.to_csv("correlation_matrix_eda.csv")


# COMMAND ----------

df_sep_3m_eda.columns

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql import functions as F

plot_columns = [
    # 'DAY_OF_MONTH',
#  'DAY_OF_WEEK',
 'OP_UNIQUE_CARRIER',
#  'OP_CARRIER_AIRLINE_ID',
 'DISTANCE_GROUP',
#  'F_MONTH',
 'is_tail_num_in_air_prev_2_hrs',
 'is_prev_flight_delayed',
#  'WeatherCode_*',
#  'WeatherCode_+DZ',
#  'WeatherCode_+FZ',
#  'WeatherCode_+PL',
#  'WeatherCode_+RA',
#  'WeatherCode_+SN',
#  'WeatherCode_-DZ',
#  'WeatherCode_-FZ',
#  'WeatherCode_-GS',
#  'WeatherCode_-PL',
#  'WeatherCode_-RA',
#  'WeatherCode_-SH',
#  'WeatherCode_-SN',
#  'WeatherCode_BC',
#  'WeatherCode_BL',
#  'WeatherCode_BLSN',
#  'WeatherCode_BR',
#  'WeatherCode_DU',
#  'WeatherCode_DZ',
#  'WeatherCode_FG',
#  'WeatherCode_FU',
#  'WeatherCode_FZ',
#  'WeatherCode_FZDZ',
#  'WeatherCode_FZRA',
#  'WeatherCode_GS',
#  'WeatherCode_HAIL',
#  'WeatherCode_HZ',
#  'WeatherCode_IC',
#  'WeatherCode_MI',
#  'WeatherCode_PL',
#  'WeatherCode_PR',
#  'WeatherCode_RA',
#  'WeatherCode_SA',
#  'WeatherCode_SH',
#  'WeatherCode_SHRA',
#  'WeatherCode_SHSN',
#  'WeatherCode_SN',
#  'WeatherCode_SQ',
#  'WeatherCode_TS',
#  'WeatherCode_UP',
#  'WeatherCode_VCBL',
#  'WeatherCode_VCFG',
#  'WeatherCode_VCTS',
#  'SkyCode_*',
#  'SkyCode_BKN',
#  'SkyCode_CLR',
#  'SkyCode_FEW',
#  'SkyCode_OVC',
#  'SkyCode_SCT',
#  'SkyCode_VV',
#  'SkyCode_X',
 'Holiday',
 'Rank',
 'pagerank']

def plot_delay_by_column(df, column_name, ax):
    delay_by_column = df.groupby(column_name).agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
    vals = delay_by_column.select('Delay_pct').collect()
    labels = delay_by_column.select(column_name).collect()
    
    vals = [val[0] for val in vals]
    labels = [label[0] for label in labels]  
    
    if len(labels) < 30:
    # Generate numerical indices for the x-axis
        x_indices = range(len(labels))
        
        ax.bar(x_indices, vals, color='green')
        ax.set_xlabel(column_name)
        ax.set_ylabel('Delay Rate')
        ax.set_title("Delay Rate by {}".format(column_name))
        
        # Set the tick labels to your categorical values
        ax.set_xticks(x_indices)
        ax.set_xticklabels(labels, rotation=90)
    else:
        x_indices = range(len(labels))
        
        ax.bar(x_indices, vals, color='green')
        ax.set_xlabel(column_name)
        ax.set_ylabel('Delay Rate')
        ax.set_title("Delay Rate by {}".format(column_name))
        
        # Set the tick labels to your categorical values
        #ax.set_xticks(x_indices)
        #ax.set_xticklabels(labels, rotation=90)

# Create subplots with appropriate layout
fig, axes = plt.subplots(4, 2, figsize=(15, 30))
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Loop through each column and plot delay rate
for i, column_name in enumerate(plot_columns):
    if column_name != "DEP_DEL15" and i<75:  # Skip the column used for aggregation
        plot_delay_by_column(df_otpw_60m_with_ranks, column_name, axes[i])

# Adjust layout
plt.tight_layout(pad=3.5)
plt.show()

# COMMAND ----------

##### Delays by DoW
delay_by_DoW = df_sep_3m_eda.groupby("is_tail_num_in_air_prev_2_hrs").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
## Delay by Day of Week Graph
vals = delay_by_DoW.select(delay_by_DoW[1]).collect()
DoW = delay_by_DoW.select(delay_by_DoW[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
DoW = [d[0] for d in DoW]
plt.bar(DoW, vals, color='Black')
plt.xlabel('Is Tail Num in Air')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_Tail_Num_in_air")
display(plt.show())

# COMMAND ----------

##### Delays by DoW
delay_by_DoW = df_sep_3m_eda.groupby("is_prev_flight_delayed").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("Delay_pct")
## Delay by Day of Week Graph
vals = delay_by_DoW.select(delay_by_DoW[1]).collect()
DoW = delay_by_DoW.select(delay_by_DoW[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
DoW = [d[0] for d in DoW]
plt.bar(DoW, vals, color='Black')
plt.xlabel('Is_Previous_Flight_Delayed')
plt.ylabel('Delay Rate')
plt.title("Delay_rate_by_Prev_Flight_Delayed")
display(plt.show())

# COMMAND ----------

# fig, axs = plt.subplots(0, 1, figsize=(50, 50))
plt.hist(df_sep_3m_eda.select('is_tail_num_in_air_prev_2_hrs').toPandas())
# axs[0,1].hist(df_sep_3m_eda.select('is_prev_flight_delayed').toPandas())
plt.title('is_tail_num_in_air_prev_2_hrs')
# axs[0,1].title.set_text('is_prev_flight_delayed')
plt.show()

# COMMAND ----------

# plt.hist(df_sep_3m_eda.select('is_tail_num_in_air_prev_2_hrs').toPandas())
plt.hist(df_sep_3m_eda.select('is_prev_flight_delayed').toPandas())
# plt.title('is_tail_num_in_air_prev_2_hrs')
plt.title('is_prev_flight_delayed')
plt.show()

# COMMAND ----------

num_rows = df_sep_3m_eda.count()  # Get the number of rows
num_cols = len(df_sep_3m_eda.columns)  # Get the number of columns

print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

# COMMAND ----------

df_sep_1y_eda = spark.read.parquet(f"{team_blob_url}/1year_cleaned")

# COMMAND ----------

num_rows = df_sep_1y_eda.count()  # Get the number of rows
num_cols = len(df_sep_1y_eda.columns)  # Get the number of columns

print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

# COMMAND ----------

