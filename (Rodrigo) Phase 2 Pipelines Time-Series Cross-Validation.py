# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

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

# MAGIC %md
# MAGIC ## Load Data from Blob Storage

# COMMAND ----------

df_sep_1y = spark.read.parquet(f"{team_blob_url}/1year_cleaned")

df_sep_1y.columns

# COMMAND ----------

df_sep_1y = df_sep_1y.drop(
 'OP_CARRIER_AIRLINE_ID',
 'ORIGIN_AIRPORT_ID',
 'DEST_AIRPORT_ID')

# COMMAND ----------

df_sep_1y = df_sep_1y.withColumn('OP_CARRIER_FL_NUM', col('OP_CARRIER_FL_NUM').cast('double'))

# COMMAND ----------

df_sep_1y.printSchema()

# COMMAND ----------

# Setup walk-forward validation.
splits = list()
curr_train_months = list()

for i in range(1, 12):
    curr_train_months.append(i)
    splits.append({'train_months': curr_train_months,
                   'test_month': i+1,
                   'train_data': df_sep_1y.filter(df_sep_1y.MONTH <= i),
                   'test_data': df_sep_1y.filter(df_sep_1y.MONTH == i+1)
                   })
splits

# COMMAND ----------

def initialize_pipeline_steps(train):
    categorical_cols = [field for (field, dataType) in train.dtypes if dataType == "string"]
    index_output_cols = [x + "Index" for x in categorical_cols]
    string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

    one_hot_vec = [x + "Vec" for x in categorical_cols]
    one_hot_encode = OneHotEncoder(inputCols=index_output_cols, outputCols=one_hot_vec) 

    numeric_cols = [field for (field, dataType) in train.dtypes if ((dataType == "double") & (field != "DEP_DEL15"))]
    imputer = Imputer(inputCols = numeric_cols, outputCols = numeric_cols)

    one_hot_cols = ['is_tail_num_in_air_prev_2_hrs', 'is_prev_flight_delayed', 'WeatherCode_*', 'WeatherCode_+DZ', 'WeatherCode_+FZ', 
                    'WeatherCode_+PL', 'WeatherCode_+RA', 'WeatherCode_+SN', 'WeatherCode_-DZ', 'WeatherCode_-FZ', 'WeatherCode_-GS', 
                    'WeatherCode_-PL', 'WeatherCode_-RA', 'WeatherCode_-SH', 'WeatherCode_-SN', 'WeatherCode_BC', 'WeatherCode_BL', 
                    'WeatherCode_BLSN', 'WeatherCode_BR', 'WeatherCode_DU','WeatherCode_DZ', 'WeatherCode_FG', 'WeatherCode_FU', 'WeatherCode_FZ', 
                    'WeatherCode_FZDZ', 'WeatherCode_FZRA', 'WeatherCode_GS', 'WeatherCode_HAIL', 'WeatherCode_HZ', 'WeatherCode_IC', 
                    'WeatherCode_MI', 'WeatherCode_PL', 'WeatherCode_PR', 'WeatherCode_RA', 'WeatherCode_SA', 'WeatherCode_SH', 'WeatherCode_SHRA', 
                    'WeatherCode_SHSN', 'WeatherCode_SN', 'WeatherCode_SQ', 'WeatherCode_TS', 'WeatherCode_UP', 'WeatherCode_VCBL', 
                    'WeatherCode_VCFG', 'WeatherCode_VCTS', 'SkyCode_*', 'SkyCode_BKN', 'SkyCode_CLR', 'SkyCode_FEW', 'SkyCode_OVC', 'SkyCode_SCT', 
                    'SkyCode_VV', 'SkyCode_X']

    assembler_inputs =  one_hot_vec + numeric_cols + one_hot_cols
    vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="vectorized_features", handleInvalid="skip")

    scaler = MinMaxScaler(inputCol="vectorized_features", outputCol="features")

    return string_indexer, one_hot_encode, imputer, vec_assembler, scaler

def initialize_evalution():
    evaluator = MulticlassClassificationEvaluator(metricName='logLoss')
    evaluatorF1 = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction", metricName="f1")
    evaluatorAcc = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction", metricName="accuracy")
    evaluatorPre = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction",metricName="weightedPrecision")
    evaluatorRec = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction", metricName="weightedRecall")
    return evaluator, evaluatorF1, evaluatorAcc, evaluatorPre, evaluatorRec

def run_evaluation(results):
    evaluator, evaluatorF1, evaluatorAcc, evaluatorPre, evaluatorRec = initialize_evalution()
    return {'f1': evaluatorF1.evaluate(results), 
            'accuracy': evaluatorAcc.evaluate(results),
            'precision': evaluatorPre.evaluate(results),
            'recall': evaluatorRec.evaluate(results)
            }

def run_iteration_of_train_and_evaluation(train, test, model_name):

    string_indexer, one_hot_encode, imputer, vec_assembler, scaler = initialize_pipeline_steps(train)

    if model_name == 'lr':
        model = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "features",labelCol='DEP_DEL15')
        model_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, scaler, model])
        
    if model_name == 'rf':
        model = RandomForestClassifier(labelCol="DEP_DEL15", featuresCol="vectorized_features", maxBins=40, seed=42)
        model_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, model])

    fitted_model = model_pipeline.fit(train) 
    results = fitted_model.transform(test)

    evaluations = run_evaluation(results)

    return evaluations

# COMMAND ----------

# Running iterations of walk-forward validation.
splits_results = splits.copy()
for i, split in enumerate(splits):
    splits_results[i]['lr_results'] = run_iteration_of_train_and_evaluation(train=splits_results[i]['train_data'], 
                                                                            test=splits_results[i]['test_data'], 
                                                                            model_name='lr')
    
    splits_results[i]['rf_results'] = run_iteration_of_train_and_evaluation(train=splits_results[i]['train_data'], 
                                                                            test=splits_results[i]['test_data'], 
                                                                            model_name='rf')

# COMMAND ----------

splits_results

# COMMAND ----------

splits_results_df = pd.DataFrame(splits_results)
splits_results_df.to_csv('splits_results.csv', index=False)

# COMMAND ----------

import pandas as pd
import ast
splits_results_df_test = pd.read_csv('splits_results.csv')
splits_results_df_test

# COMMAND ----------

def get_metric(metric, res_df, model_name):
    return splits_results_df_test[model_name + '_results'].apply(lambda x: ast.literal_eval(x)[metric]).values    

metrics = ['accuracy', 'precision', 'recall', 'f1']

lr_accuracies, lr_precisions, lr_recalls, lr_f1s = map(lambda metric: get_metric(metric=metric, 
                                                                                 res_df=splits_results_df_test, 
                                                                                 model_name='lr'), metrics)

rf_accuracies, rf_precisions, rf_recalls, rf_f1s = map(lambda metric: get_metric(metric=metric, 
                                                                                 res_df=splits_results_df_test, 
                                                                                 model_name='rf'), metrics)

# COMMAND ----------

import matplotlib.pyplot as plt

x = list(range(2, 13))

plt.plot(x, rf_accuracies, label='accuracy',color='yellow')
plt.plot(x, rf_precisions, label='precision', linestyle='dashed')
plt.plot(x, rf_recalls, label='recall', linestyle='dashed', color = 'black')
plt.plot(x, rf_f1s, label='f1', linestyle='dashed')

plt.legend()
plt.title('Evaluation metrics for Random Forest Cross-Validation')
plt.xlabel('Evaluation Month (Training on all previous months in year)')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

x = list(range(2, 13))

plt.plot(x, lr_accuracies, label='accuracy',color='yellow')
plt.plot(x, lr_precisions, label='precision', linestyle='dashed')
plt.plot(x, lr_recalls, label='recall', linestyle='dashed', color = 'black')
plt.plot(x, lr_f1s, label='f1', linestyle='dashed')

plt.legend()
plt.title('Evaluation metrics for Logistic Regression Cross-Validation')
plt.xlabel('Evaluation Month (Training on all previous months in year)')
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

