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
display(df_otpw_60m)

# COMMAND ----------

df_otpw_60m = spark.read.parquet(f"{team_blob_url}/60month_cleaned_new_1")
#df_otpw_60m

# COMMAND ----------

df_otpw_60m_2015 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2015)
df_otpw_60m_2016 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2016)
df_otpw_60m_2017 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2017)
df_otpw_60m_2018 = df_otpw_60m.filter(df_otpw_60m.YEAR<=2018) 

# COMMAND ----------

from pyspark.sql.functions import col
from graphframes import *
from pyspark.sql import functions as F

def save_to_disk(df, filepath):
    df.write.mode('overwrite').parquet(f"{team_blob_url}/{filepath}".format(filepath=filepath))

def get_PageRanks(df, write=False, filename=None):
    v = df.select(col("ORIGIN_AIRPORT_ID").alias("id")).distinct()
    e = df.select(col("ORIGIN_AIRPORT_ID").alias("src"), col("DEST_AIRPORT_ID").alias("dst"))
    g = GraphFrame(v, e)

    # call the pageRank method, with resetProbability=0.15 and maxIter=10
    results = g.pageRank(resetProbability=0.15, maxIter=10)

    if write:
        if filename is None:
            print('Please specify file name. Could not save to disk.')
        else:
            save_to_disk(results.vertices, 'pageranks/{filename}'.format(filename=filename))

    return results

# COMMAND ----------

# Calculate Pageranks
ranks_2015 = get_PageRanks(df=df_otpw_60m_2015, write=True, filename='_2015')
ranks_2016 = get_PageRanks(df=df_otpw_60m_2016, write=True, filename='_2016')
ranks_2017 = get_PageRanks(df=df_otpw_60m_2017, write=True, filename='_2017')
ranks_2018 = get_PageRanks(df=df_otpw_60m_2018, write=True, filename='_2018')

# COMMAND ----------

# Load Pagerank info (so that calculations don't need to be run again.)
pageranks_2015 = spark.read.parquet(f"{team_blob_url}/pageranks/_2015")
pageranks_2016 = spark.read.parquet(f"{team_blob_url}/pageranks/_2016")
pageranks_2017 = spark.read.parquet(f"{team_blob_url}/pageranks/_2017")
pageranks_2018 = spark.read.parquet(f"{team_blob_url}/pageranks/_2018")

# COMMAND ----------

# Visuals
pageranks_2018_viz = pageranks_2018.join(other=df_otpw_60m_2018.select('ORIGIN_AIRPORT_ID', 'ORIGIN').dropDuplicates(),
                                         on= pageranks_2018.id == df_otpw_60m_2018.ORIGIN_AIRPORT_ID, 
                                         how='left').select('ORIGIN', 'pagerank')
pageranks_2018_viz

# COMMAND ----------

# Select the top 10 records sorted by pagerank
top_10 = pageranks_2018_viz.orderBy(pageranks_2018_viz.pagerank.desc()).limit(10).toPandas()

# Select the bottom 10 records sorted by pagerank
bottom_10 = pageranks_2018_viz.orderBy(pageranks_2018_viz.pagerank).limit(10).toPandas()

# Select all airports pagerank
all_airports = pageranks_2018_viz.toPandas()


# COMMAND ----------

import matplotlib.pyplot as plt

# Plot the top 10 airports
plt.figure(figsize=(10, 6))
plt.bar(top_10['ORIGIN'], top_10['pagerank'], color='darkgreen')
plt.xlabel('Airport')
plt.ylabel('Pagerank')
plt.title('Top 10 Airports by Pagerank')
plt.xticks(rotation=45)
plt.show()

# Plot the bottom 10 airports
plt.figure(figsize=(10, 6))
plt.bar(bottom_10['ORIGIN'], bottom_10['pagerank'], color='green')
plt.xlabel('Airport')
plt.ylabel('Pagerank')
plt.title('Bottom 10 Airports by Pagerank')
plt.xticks(rotation=45)
plt.show()

# Plot all of the airports
plt.figure(figsize=(10, 6))
plt.bar(all_airports['ORIGIN'], all_airports['pagerank'], color='lightgreen')
plt.xlabel('Airport')
plt.ylabel('Pagerank')
plt.title('Pageranks By Airport')
plt.xticks(rotation=90, fontsize=2)
plt.show()

# Plot hist all pageranks
plt.figure(figsize=(10, 6))
plt.hist(all_airports['pagerank'], bins=1000, color='lightgreen')
plt.xlabel('Pagerank')
plt.ylabel('Frequency')
plt.title('Distribution of Pageranks')
plt.xticks(rotation=0)
plt.show()

# COMMAND ----------

