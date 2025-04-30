# Databricks notebook source
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

# Read the CSV files into DataFrames
jan_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jan_2020.csv")
jan_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jan_2021.csv")
jan_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jan_2022.csv")
feb_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Feb_2020.csv")
feb_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Feb_2021.csv")
feb_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Feb_2022.csv")
mar_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Mar_2020.csv")
mar_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Mar_2021.csv")
mar_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Mar_2022.csv")
apr_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jan_2020.csv")
apr_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Apr_2021.csv")
apr_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Apr_2022.csv")
may_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/May_2020.csv")
may_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/May_2021.csv")
may_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/May_2022.csv")
jun_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jun_2020.csv")
jun_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jun_2021.csv")
jun_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jun_2022.csv")
jul_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jul_2020.csv")
jul_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jul_2021.csv")
jul_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Jul_2022.csv")
aug_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Aug_2020.csv")
aug_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Aug_2021.csv")
aug_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Aug_2022.csv")
sep_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Sep_2020.csv")
sep_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Sep_2021.csv")
sep_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Sep_2022.csv")
oct_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Oct_2020.csv")
oct_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Oct_2021.csv")
oct_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Oct_2022.csv")
nov_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Nov_2020.csv")
nov_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Nov_2021.csv")
nov_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Nov_2022.csv")
dec_2020 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Dec_2020.csv")
dec_2021 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Dec_2021.csv")
dec_2022 = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/New_Flight_Data/Dec_2022.csv")

# COMMAND ----------

may_2021 = may_2021.drop('FL_DATE')
jun_2021 = jun_2021.drop('FL_DATE')
jul_2021 = jul_2021.drop('FL_DATE')
aug_2021 = aug_2021.drop('FL_DATE')
sep_2021 = sep_2021.drop('FL_DATE')
oct_2021 = oct_2021.drop('FL_DATE')
nov_2021 = nov_2021.drop('FL_DATE')
jan_2022 = jan_2022.drop('FL_DATE')
feb_2022 = feb_2022.drop('FL_DATE')
mar_2022 = mar_2022.drop('FL_DATE')
apr_2022 = apr_2022.drop('FL_DATE')
may_2022 = may_2022.drop('FL_DATE')
jun_2022 = jun_2022.drop('FL_DATE')
jul_2022 = jul_2022.drop('FL_DATE')
aug_2022 = aug_2022.drop('FL_DATE')
sep_2022 = sep_2022.drop('FL_DATE')
oct_2022 = oct_2022.drop('FL_DATE')
nov_2022 = nov_2022.drop('FL_DATE')
# dec_2022 = dec_2022.drop('FL_DATE')

# COMMAND ----------

dec_2022.display()

# COMMAND ----------

## Combine into one df
from functools import reduce
from pyspark.sql import DataFrame
dfs = [jan_2020, jan_2021, jan_2022, feb_2020, mar_2020, apr_2020, may_2020, jun_2020, jul_2020, aug_2020, sep_2020, oct_2020, nov_2020, dec_2020, feb_2021, mar_2021, apr_2021, may_2021, jun_2021, jul_2021, aug_2021, sep_2021, oct_2021, nov_2021, dec_2021, feb_2022, mar_2022, apr_2022, may_2022, jun_2022, jul_2022, aug_2022, sep_2022, oct_2022, nov_2022, dec_2022]
df_flight_new = reduce(DataFrame.unionAll, dfs)
# display(df_otpw_60m)

# COMMAND ----------

df_flight_new = df_flight_new.filter(df_flight_new.DEP_DEL15.isNotNull())

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
df_stg = df_flight_new.withColumn(
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
df_flight_new = df_stg.withColumn(
    'is_prev_flight_delayed',
    when((lag_window_delay.cast("double") == 1)
               &
      (lag_window_DoM == F.col("DAY_OF_MONTH")), 1).otherwise(0)

)

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
    df_flight_new = df_flight_new.withColumn(column, col(column).cast('double'))

# COMMAND ----------

df_flight_new = df_flight_new.na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

# Read the CSV files into DataFrames
rankings = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/WSJ_Rankings.csv")
holidays = spark.read.option("header", "true").csv("file:/Workspace/Users/rodrigocochran@berkeley.edu/w261-final_project-team-2_1/Holidays.csv")

# COMMAND ----------

# Join tables for Rankings
df_flight_new = df_flight_new.join(rankings, (df_flight_new.OP_UNIQUE_CARRIER == rankings.Carrier) & (df_flight_new.YEAR == rankings.Ranking_YEAR), how='left')

# Handle None Rank values
df_flight_new = df_flight_new.fillna({ 'Rank': 10 })

# Join tables for Holidays
df_flight_new = df_flight_new.join(holidays, (df_flight_new.DAY_OF_MONTH == holidays.Day) & (df_flight_new.MONTH == holidays.Holiday_Month) & (df_flight_new.YEAR == holidays.Holiday_Year), how='left')

# COMMAND ----------

## Delete redundant columns
df_flight_new =  df_flight_new.drop('Full Name', 'Carrier', 'Date', 'WeekDay', 'Day', 'Holiday_Year', 'Holiday_Month', 'Ranking_YEAR')

# COMMAND ----------

df_flight_new.display()

# COMMAND ----------

df_flight_new.write.parquet(f"{team_blob_url}/new_flight_data_1")

# COMMAND ----------

