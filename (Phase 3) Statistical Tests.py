# Databricks notebook source
from pyspark.sql.functions import col
print("Welcome to the W261 final project!")

# COMMAND ----------

import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

# COMMAND ----------

## Place this cell in any team notebook that needs access to the team cloud storage.


# The following blob storage is accessible to team members only (read and write)
# access key is valid til TTL
# after that you will need to create a new SAS key and authenticate access again via DataBrick command line
blob_container  = "data"       # The name of your container created in https://portal.azure.com
storage_account = "rodrigocochran"  # The name of your Storage account created in https://portal.azure.com
secret_scope    = "team_2_1"           # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "team_2_1_key"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket


# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)
import pandas as pd
pdf = pd.DataFrame([[1, 2, 3, "Jane"], [2, 2,2, None], [12, 12,12, "John"]], columns=["x", "y", "z", "a_string"])
df = spark.createDataFrame(pdf) # Create a Spark dataframe from a pandas DF

# The following can write the dataframe to the team's Cloud Storage  
# Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
#df.write.parquet(f"{team_blob_url}/TP_rodrigo")



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
                                               on= df_otpw_60m_2018.ORIGIN_AIRPORT_ID == pageranks_2018.id, 
                                               how='left')
df_otpw = df_otpw_60m_with_ranks

# COMMAND ----------

# MAGIC %md
# MAGIC # Statistical Tests

# COMMAND ----------

from pyspark.ml.stat import ChiSquareTest
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delay by Day of the week

# COMMAND ----------

dow_delay_df = df_otpw.select(['DAY_OF_WEEK', 'DEP_DEL15'])
dow_delay_df = dow_delay_df.dropna(subset='DEP_DEL15')
dow_delay_df = dow_delay_df.withColumn("DAY_OF_WEEK", dow_delay_df["DAY_OF_WEEK"].cast(IntegerType()))
dow_delay_df = dow_delay_df.withColumn("DEP_DEL15", dow_delay_df["DEP_DEL15"].cast(IntegerType()))

assembler = VectorAssembler().setInputCols(['DAY_OF_WEEK']).setOutputCol('features')

result = ChiSquareTest.test(dataset=assembler.transform(dow_delay_df), featuresCol="features", labelCol="DEP_DEL15", flatten=True)
row = result.orderBy("featureIndex").collect()

print('Degrees of freedom: ', row[0].degreesOfFreedom)
print('Statistic:', row[0].statistic)
print('P-value:', row[0].pValue)

# COMMAND ----------

# MAGIC %md
# MAGIC This Chi-Square test will measure the independence of the days of the week based on delays. \
# MAGIC In our case we will be looking at the DAY_OF_WEEK variable, which has 6 degrees of freedom, and we are using alpha = .05. \
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the DAY_OF_WEEK variable. \
# MAGIC In layman's terms, there is statistical evidence to believe that the day of the week affects if a plane will be delayed by 15 or more minutes.

# COMMAND ----------

delay_by_DoW = df_otpw.groupby("DAY_OF_WEEK").agg(F.avg("DEP_DEL15").alias('Delay_pct')).orderBy("DAY_OF_WEEK")
display(delay_by_DoW)


day_of_week_map = {'1': 'Mon',
                   '2': 'Tues',
                   '3': 'Weds',
                   '4': 'Thurs',
                   '5': 'Fri',
                   '6': 'Sat',
                   '7': 'Sun',
                    }

## Delay by Day of Week Graph
vals = delay_by_DoW.select(delay_by_DoW[1]).collect()
DoW = delay_by_DoW.select(delay_by_DoW[0]).collect()
# Convert numpy arrays to lists
vals = [val[0] for val in vals]
DoW = [day_of_week_map[str(d[0]).split('.')[0]] for d in DoW]
plt.bar(DoW, vals)
plt.xlabel('Day of Week')
plt.ylabel('Delay Rate')
plt.title("Delay Rate by Day of the Week")
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delay by Distance

# COMMAND ----------

from math import sqrt
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

def independent_ttest(data1, data2, alpha, col_name, verbose=False):
    # calculate means
    mean1 = data1.agg(F.mean(data1[col_name]).alias('mean1')).collect()[0]['mean1']
    mean2 = data2.agg(F.mean(data2[col_name]).alias('mean2')).collect()[0]['mean2']
    
    # calculate standard errors
    se1 = data1.agg((F.stddev(data1[col_name]) / F.sqrt(F.count(data1[col_name]))).alias('se1')).collect()[0]['se1']
    se2 = data2.agg((F.stddev(data2[col_name]) / F.sqrt(F.count(data2[col_name]))).alias('se2')).collect()[0]['se2']
    
    # standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
    
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    
    # degrees of freedom
    df = data1.count() + data2.count() - 2
    
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    
    # return everything
    if verbose:
        print("T-statistic:", t_stat)
        print("Degrees of freedom:", df)
        print("Critical value:", cv)
        print("P-value:", p)
    return t_stat, df, cv, p

# COMMAND ----------

distance_delay_df = df_otpw.select(['DISTANCE', 'DEP_DEL15'])
distance_delay_df = distance_delay_df.dropna(subset='DEP_DEL15')
distance_delay_df = distance_delay_df.withColumn("DISTANCE", distance_delay_df["DISTANCE"].cast(IntegerType()))
distance_delay_df = distance_delay_df.withColumn("DEP_DEL15", distance_delay_df["DEP_DEL15"].cast(IntegerType()))

distance_delay = distance_delay_df.filter("DEP_DEL15 = 1").select("DISTANCE")
distance_no_delay = distance_delay_df.filter("DEP_DEL15 = 0").select("DISTANCE")
t_stat, df, cv, p = independent_ttest(data1=distance_delay, data2=distance_no_delay, alpha=0.05, col_name = 'DISTANCE', verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC This two-sample t-test of independence aims to disprove the null hypothesis that delays (of 15 or more minutes) are independent of the distance of the flight. \
# MAGIC In this test we are analyzing the DISTANCE variable divided into two groups (delay or not delay) based on the DEP_DEL15 variable. \
# MAGIC Again, we are using alpha = 0.05 as a threshold for rejecting the null hypothesis. In this case, the test yielded a value of 0.0 and therefore we can reject the null hypothesis. \ 
# MAGIC In layman's terms, this means that there is statistically significant evidience to believe that the distance of the flight affects whether the flight will be delayed. 

# COMMAND ----------

distance_delay = distance_delay_df.filter("DEP_DEL15 = 1").select("DISTANCE").collect()
distance_no_delay = distance_delay_df.filter("DEP_DEL15 = 0").select("DISTANCE").collect()
distance_no_delay_list =  [row['DISTANCE'] for row in distance_no_delay]
distance_delay_list =  [row['DISTANCE'] for row in distance_delay]
plt.hist(distance_no_delay_list, bins=100, alpha=0.5, label='No Delay')
plt.hist(distance_delay_list,  bins=100, alpha=0.5, label='Delay')
plt.legend(loc='upper right')
plt.title('Distribution of Flight Distances by Delay (or No Delay)')
plt.xlabel('Flight Distance')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delay by Origin

# COMMAND ----------

origin_delay_df = df_otpw.select(['ORIGIN_AIRPORT_ID', 'DEP_DEL15'])
origin_delay_df = origin_delay_df.dropna(subset='DEP_DEL15')
origin_delay_df = origin_delay_df.withColumn("ORIGIN_AIRPORT_ID", origin_delay_df["ORIGIN_AIRPORT_ID"].cast(IntegerType()))
origin_delay_df = origin_delay_df.withColumn("DEP_DEL15", origin_delay_df["DEP_DEL15"].cast(IntegerType()))

assembler = VectorAssembler().setInputCols(['ORIGIN_AIRPORT_ID']).setOutputCol('features')

result = ChiSquareTest.test(dataset=assembler.transform(origin_delay_df), featuresCol="features", labelCol="DEP_DEL15", flatten=True)
row = result.orderBy("featureIndex").collect()

print('Degrees of freedom: ', row[0].degreesOfFreedom)
print('Statistic:', row[0].statistic)
print('P-value:', row[0].pValue)

# COMMAND ----------

# MAGIC %md
# MAGIC This Chi-Square test will measure the independence of the origin airports based on delays. \
# MAGIC In our case we will be looking at the ORIGIN_AIRPORT_ID variable, which has 312 degrees of freedom, and we are using alpha = .05. \
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the ORIGIN_AIRPORT_ID variable. \
# MAGIC In layman's terms, there is statistical evidence to believe that the origin airport affects if a plane will be delayed by 15 or more minutes.

# COMMAND ----------

import matplotlib.pyplot as plt

# Group df_otpw by origin and calculate the average of DEP_DEL15
origin_avg_delay = df_otpw.groupBy("ORIGIN").avg("DEP_DEL15")

# Sort the data by average delay in ascending order
origin_avg_delay = origin_avg_delay.orderBy("avg(DEP_DEL15)")

# Extract the origin airport IDs and average delay values for the top 10 and bottom 10
top_10_origin_ids = [row["ORIGIN"] for row in origin_avg_delay.tail(10)]
top_10_avg_delays = [row["avg(DEP_DEL15)"] for row in origin_avg_delay.tail(10)]

bottom_10_origin_ids = [row["ORIGIN"] for row in origin_avg_delay.head(10)]
bottom_10_avg_delays = [row["avg(DEP_DEL15)"] for row in origin_avg_delay.head(10)]

# Create a bar chart to visualize the average delays for the top 10 origins
plt.figure(figsize=(10,4))
plt.bar(top_10_origin_ids, top_10_avg_delays, color='red')
plt.xlabel("Origin Airport")
plt.ylabel("Delay Rate")
plt.title("Top 10 Highest Delay Rates by Origin Airport")
plt.show()

# Create a bar chart to visualize the average delays for the bottom 10 origins
plt.figure(figsize=(10,4))
plt.bar(bottom_10_origin_ids, bottom_10_avg_delays, color='pink')
plt.xlabel("Origin Airport")
plt.ylabel("Delay Rate")
plt.title("Top 10 Lowest Delay Rates by Origin Airport")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delay by Destination

# COMMAND ----------

dest_delay_df = df_otpw.select(['DEST_AIRPORT_ID', 'DEP_DEL15'])
dest_delay_df = dest_delay_df.dropna(subset='DEP_DEL15')
dest_delay_df = dest_delay_df.withColumn("DEST_AIRPORT_ID", dest_delay_df["DEST_AIRPORT_ID"].cast(IntegerType()))
dest_delay_df = dest_delay_df.withColumn("DEP_DEL15", dest_delay_df["DEP_DEL15"].cast(IntegerType()))

assembler = VectorAssembler().setInputCols(['DEST_AIRPORT_ID']).setOutputCol('features')

result = ChiSquareTest.test(dataset=assembler.transform(dest_delay_df), featuresCol="features", labelCol="DEP_DEL15", flatten=True)
row = result.orderBy("featureIndex").collect()

print('Degrees of freedom: ', row[0].degreesOfFreedom)
print('Statistic:', row[0].statistic)
print('P-value:', row[0].pValue)

# COMMAND ----------

# MAGIC %md
# MAGIC This Chi-Square test will measure the independence of the destination airports based on delays. \
# MAGIC In our case we will be looking at the DEST_AIRPORT_ID variable, which has 314 degrees of freedom, and we are using alpha = .05.
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the DEST_AIRPORT_ID variable.
# MAGIC In layman's terms, there is statistical evidence to believe that the destination airport affects if a plane will be delayed by 15 or more minutes.

# COMMAND ----------

import matplotlib.pyplot as plt

# Group df_otpw by dest and calculate the average of DEP_DEL15
dest_avg_delay = df_otpw.groupBy("DEST").avg("DEP_DEL15")

# Sort the data by average delay in ascending order
dest_avg_delay = dest_avg_delay.orderBy("avg(DEP_DEL15)")

# Extract the dest airport IDs and average delay values for the top 10 and bottom 10
top_10_dest_ids = [row["DEST"] for row in dest_avg_delay.tail(10)]
top_10_avg_delays = [row["avg(DEP_DEL15)"] for row in dest_avg_delay.tail(10)]

bottom_10_dest_ids = [row["DEST"] for row in dest_avg_delay.head(10)]
bottom_10_avg_delays = [row["avg(DEP_DEL15)"] for row in dest_avg_delay.head(10)]

# Create a bar chart to visualize the average delays for the top 10 dests
plt.figure(figsize=(10,4))
plt.bar(top_10_dest_ids, top_10_avg_delays, color='orange')
plt.xlabel("Destination Airport")
plt.ylabel("Delay Rate")
plt.title("Top 10 Highest Delay Rates by Destination Airport")
plt.show()

# Create a bar chart to visualize the average delays for the bottom 10 dests
plt.figure(figsize=(10,4))
plt.bar(bottom_10_dest_ids, bottom_10_avg_delays, color='purple')
plt.xlabel("Destination Airport")
plt.ylabel("Delay Rate")
plt.title("Top 10 Lowest Delay Rates by Desination Airport")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Delay by Pagerank

# COMMAND ----------

pagerank_delay_df = df_otpw.select(['pagerank', 'DEP_DEL15'])
pagerank_delay_df = pagerank_delay_df.dropna(subset='DEP_DEL15')
pagerank_delay_df = pagerank_delay_df.withColumn("pagerank", pagerank_delay_df["pagerank"].cast(DoubleType()))
pagerank_delay_df = pagerank_delay_df.withColumn("DEP_DEL15", pagerank_delay_df["DEP_DEL15"].cast(IntegerType()))

pagerank_delay = pagerank_delay_df.filter("DEP_DEL15 = 1").select("pagerank")
pagerank_no_delay = pagerank_delay_df.filter("DEP_DEL15 = 0").select("pagerank")
t_stat, df, cv, p = independent_ttest(data1=pagerank_delay, data2=pagerank_no_delay, alpha=0.05, col_name = 'pagerank', verbose=True)

# COMMAND ----------

pagerank_delay = pagerank_delay_df.filter("DEP_DEL15 = 1").select("pagerank").collect()
pagerank_no_delay = pagerank_delay_df.filter("DEP_DEL15 = 0").select("pagerank").collect()
pagerank_no_delay_list =  [row['pagerank'] for row in pagerank_no_delay]
pagerank_delay_list =  [row['pagerank'] for row in pagerank_delay]


# COMMAND ----------

import matplotlib.pyplot as plt
plt.hist(pagerank_no_delay_list, bins=100, alpha=0.5, label='No Delay')
plt.hist(pagerank_delay_list,  bins=100, alpha=0.5, label='Delay')
plt.legend(loc='upper right')
plt.title('Distribution of Pageranks by Delay (or No Delay)')
plt.xlabel('Pagerank')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

