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

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer, MinMaxScaler, OneHotEncoder, PCA
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier, GBTClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pyspark.sql.functions as F
from pyspark.sql.functions import col, size


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
# MAGIC # Load Source Data

# COMMAND ----------

df_60m = spark.read.parquet(f"{team_blob_url}/60month_cleaned_new_1")

# COMMAND ----------

df_60m = df_60m.drop('OP_CARRIER_AIRLINE_ID','DEST_AIRPORT_ID','OP_CARRIER_FL_NUM')

# COMMAND ----------

df_60m  = df_60m.withColumnRenamed("DEP_DEL15", "label")
df_60m  = df_60m.withColumn('label', col('label').cast('int'))

# COMMAND ----------

df_60m = df_60m.withColumn('DAY_OF_MONTH', col('DAY_OF_MONTH').cast('string'))
df_60m = df_60m.withColumn('DAY_OF_WEEK', col('DAY_OF_WEEK').cast('string'))
df_60m = df_60m.withColumn('F_MONTH', col('F_MONTH').cast('string'))

# COMMAND ----------

# MAGIC %md
# MAGIC # Pipeline Functions

# COMMAND ----------

def get_pipeline_steps(df):
    # Create categorical cols list
    categorical_cols = [field for (field, dataType) in df.dtypes if dataType == "string"]
    # categorical_cols.remove('ORIGIN_AIRPORT_ID')

    # Pass categorcial cols through STring indexer
    index_output_cols = [x + "Index" for x in categorical_cols]
    string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

    # Pass categorical cols through one_hot_encoder
    one_hot_vec = [x + "Vec" for x in categorical_cols]
    one_hot_encode = OneHotEncoder(inputCols=index_output_cols, 
                                   outputCols=one_hot_vec) 
                                
    # Create list of already handled one hot columns
    one_hot_cols = ['is_tail_num_in_air_prev_2_hrs',
                    'is_prev_flight_delayed',
                    'WeatherCode_*',
                    'WeatherCode_+DZ',
                    'WeatherCode_+FZ',
                    'WeatherCode_+PL',
                    'WeatherCode_+RA',
                    'WeatherCode_+SN',
                    'WeatherCode_-DZ',
                    'WeatherCode_-FZ',
                    'WeatherCode_-GS',
                    'WeatherCode_-PL',
                    'WeatherCode_-RA',
                    'WeatherCode_-SH',
                    'WeatherCode_-SN',
                    'WeatherCode_BC',
                    'WeatherCode_BL',
                    'WeatherCode_BLSN',
                    'WeatherCode_BR',
                    'WeatherCode_DU',
                    'WeatherCode_DZ',
                    'WeatherCode_FG',
                    'WeatherCode_FU',
                    'WeatherCode_FZ',
                    'WeatherCode_FZDZ',
                    'WeatherCode_FZRA',
                    'WeatherCode_GS',
                    'WeatherCode_HAIL',
                    'WeatherCode_HZ',
                    'WeatherCode_IC',
                    'WeatherCode_MI',
                    'WeatherCode_PL',
                    'WeatherCode_PR',
                    'WeatherCode_RA',
                    'WeatherCode_SA',
                    'WeatherCode_SH',
                    'WeatherCode_SHRA',
                    'WeatherCode_SHSN',
                    'WeatherCode_SN',
                    'WeatherCode_SQ',
                    'WeatherCode_TS',
                    'WeatherCode_UP',
                    'WeatherCode_VCBL',
                    'WeatherCode_VCFG',
                    'WeatherCode_VCTS',
                    'SkyCode_*',
                    'SkyCode_BKN',
                    'SkyCode_CLR',
                    'SkyCode_FEW',
                    'SkyCode_OVC',
                    'SkyCode_SCT',
                    'SkyCode_VV',
                    'SkyCode_X']

    # Extract numeric columns and pass through imputer to deal with missing values
    numeric_cols = [field for (field, dataType) in df.dtypes if ((dataType == "double") & (field != "label"))]
    # numeric_cols.append('pagerank')
    imputer = Imputer(inputCols = numeric_cols, outputCols = numeric_cols)

    # Assemble vector of inputs
    assembler_inputs =  one_hot_vec + numeric_cols + one_hot_cols

    vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="vectorized_features", handleInvalid="skip")

    # Scale inputs
    scaler = MinMaxScaler(inputCol="vectorized_features", outputCol="features")

    return string_indexer, one_hot_encode, imputer, vec_assembler, scaler

# COMMAND ----------

def get_train_test_split(df, year, pageranks):
    train = df.filter(df.YEAR<=year)
    test = df.filter(df.YEAR==year+1)
    train = train.join(other=pageranks, 
                       on=train.ORIGIN_AIRPORT_ID == pageranks.id, 
                       how='left').drop('id', 'ORIGIN_AIRPORT_ID', 'OP_CARRIER_AIRLINE_ID','DEST_AIRPORT_ID','OP_CARRIER_FL_NUM','DISTANCE_GROUP','QUARTER', 'YEAR')
    test = test.join(other=pageranks, 
                       on=test.ORIGIN_AIRPORT_ID == pageranks.id, 
                       how='left').drop('id', 'ORIGIN_AIRPORT_ID', 'OP_CARRIER_AIRLINE_ID','DEST_AIRPORT_ID','OP_CARRIER_FL_NUM','DISTANCE_GROUP','QUARTER', 'YEAR')
    return train, test

def get_model_base(model_type, input_layer_size=None):
    if model_type == 'rf':
        return RandomForestClassifier(labelCol="label", featuresCol="vectorized_features", maxBins=40, seed=42)
    if model_type == 'gbt':
        return GBTClassifier(labelCol="label", featuresCol="vectorized_features", maxBins=40, seed=42)
    if model_type == 'mlp_v1':
        layers_v1 = [input_layer_size, 5, 2]
        return MultilayerPerceptronClassifier(labelCol='label',
                                                    featuresCol='features',
                                                    maxIter=10,
                                                    layers=layers_v1)
    if model_type == 'mlp_v2':
        layers_v2 = [input_layer_size, 5, 5, 2]
        return MultilayerPerceptronClassifier(labelCol='label',
                                                    featuresCol='features',
                                                    maxIter=10,
                                                    layers=layers_v2
                                                    )

def get_pipeline(model_type, train, input_layer_size=None):
    string_indexer, one_hot_encode, imputer, vec_assembler, scaler = get_pipeline_steps(train)
    if model_type == 'rf':
        return Pipeline(stages=[string_indexer, one_hot_encode, imputer, vec_assembler, get_model_base(model_type)])
    if model_type == 'gbt':
        return Pipeline(stages=[string_indexer, one_hot_encode, imputer, vec_assembler, get_model_base(model_type)])
    if model_type == 'mlp_v1':
        return Pipeline(stages=[string_indexer, one_hot_encode, imputer, vec_assembler, scaler, get_model_base(model_type,
                                                                                                               input_layer_size
                                                                                                               )])
    if model_type == 'mlp_v2':
        return Pipeline(stages=[string_indexer, one_hot_encode, imputer, vec_assembler, scaler, get_model_base(model_type,
                                                                                                               input_layer_size
                                                                                                               )])

def initialize_eval_metrics():
    # Model Evaluation Metrics
    evaluator = MulticlassClassificationEvaluator(metricName='logLoss')
    evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluatorAcc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluatorPre = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="weightedPrecision")
    evaluatorRec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    return evaluator, evaluatorF1, evaluatorAcc, evaluatorPre, evaluatorRec


def save_to_disk(df, filepath):
    df.write.mode('overwrite').parquet(f"{team_blob_url}/{filepath}".format(filepath=filepath))

def calc_evals(results):
    f1score = evaluatorF1.evaluate(results)
    accuracy = evaluatorAcc.evaluate(results)
    precision = evaluatorPre.evaluate(results)
    recall = evaluatorRec.evaluate(results)

    return {'f1score': f1score, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall}

def test_basic_model(train, test, pipeline):
    # Fitting the model on training data 
    # print(pipeline.fit(train).head())
    fit_model = pipeline.fit(train)

    # print(fit_model.explainParams())
    
    # Storing the results on test data 
    results = fit_model.transform(test)

    #return results

    return calc_evals(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Pagerank Info

# COMMAND ----------

# Load Pagerank info
pageranks_2015 = spark.read.parquet(f"{team_blob_url}/pageranks/_2015")
pageranks_2016 = spark.read.parquet(f"{team_blob_url}/pageranks/_2016")
pageranks_2017 = spark.read.parquet(f"{team_blob_url}/pageranks/_2017")
pageranks_2018 = spark.read.parquet(f"{team_blob_url}/pageranks/_2018")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Train and Test Sets

# COMMAND ----------

train_2015, test_2016 = get_train_test_split(df=df_60m, year=2015, pageranks=pageranks_2015)
train_2016, test_2017 = get_train_test_split(df=df_60m, year=2016, pageranks=pageranks_2016)
train_2017, test_2018 = get_train_test_split(df=df_60m, year=2017, pageranks=pageranks_2017)
train_2018, test_2019 = get_train_test_split(df=df_60m, year=2018, pageranks=pageranks_2018)

# COMMAND ----------

splits = ({'train_year': '2015', 'train_df': train_2015, 'test_year': '2016', 'test_df': test_2016},
          {'train_year': '2016', 'train_df': train_2016, 'test_year': '2017', 'test_df': test_2017},
          {'train_year': '2017', 'train_df': train_2017, 'test_year': '2018', 'test_df': test_2018},
          {'train_year': '2018', 'train_df': train_2018, 'test_year': '2019', 'test_df': test_2019})

# COMMAND ----------

# Initialize in-memory list to store resulst
cross_val_results = list()

# Define models to be executed for Cross-Validation
model_types = ['rf' 'gbt', 'mlp_v1', 'mlp_v2']

# Pre-defined input layer sizes (only applies to MLP models).
input_layer_sizes = {'2015': 793,
                     '2016': 800,
                     '2017': 814,
                     '2018': 884,
                     }

# Initialize evaluation metrics
evaluator, evaluatorF1, evaluatorAcc, evaluatorPre, evaluatorRec = initialize_eval_metrics()

for model_type in model_types:
    for split in splits:
        cross_val_result = {'train_year': split['train_year'],
                            'test_year': split['test_year'],
                            'model_type': model_type,
                            'results': test_basic_model(train=split['train_df'],
                                                        test=split['test_df'],
                                                        pipeline=get_pipeline(model_type=model_type, 
                                                                              train=split['train_df'],
                                                                              input_layer_size= input_layer_sizes[split['train_year']]
                                                                              ))}
        # Store after each iteration to mitigate risk of needing to re-execute code.
        filepath = 'phase_3_cv_results/'
        filepath = filepath + '{model_type}/test_year_{test_year}'.format(model_type=model_type, test_year=split['test_year'])
        save_to_disk(df=spark.createDataFrame([cross_val_result]), filepath=filepath)
        print(cross_val_result)
        cross_val_results.append(cross_val_result)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare and Plot Results

# COMMAND ----------

mlp_v1_2016_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v1/test_year_2016")
mlp_v1_2017_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v1/test_year_2017")
mlp_v1_2018_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v1/test_year_2018")
mlp_v1_2019_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v1/test_year_2019")

mlp_v2_2016_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v2/test_year_2016")
mlp_v2_2017_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v2/test_year_2017")
mlp_v2_2018_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v2/test_year_2018")
mlp_v2_2019_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/mlp_v2/test_year_2019")

# COMMAND ----------

gbt_2016_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/gbt/test_year_2016")
gbt_2017_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/gbt/test_year_2017")
gbt_2018_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/gbt/test_year_2018")
gbt_2019_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/gbt/test_year_2019")

rf_2016_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/rf/test_year_2016")
rf_2017_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/rf/test_year_2017")
rf_2018_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/rf/test_year_2018")
rf_2019_results = spark.read.parquet(f"{team_blob_url}/phase_3_cv_results/rf/test_year_2019")



# COMMAND ----------

# store results in a concatenated df to display
gbt_results = pd.concat([gbt_2016_results.toPandas(),
                         gbt_2017_results.toPandas(),
                         gbt_2018_results.toPandas(),
                         gbt_2019_results.toPandas(),
                         ])
rf_results = pd.concat([rf_2016_results.toPandas(),
                        rf_2017_results.toPandas(),
                        rf_2018_results.toPandas(),
                        rf_2019_results.toPandas(),
                        ])
mlp_v1_results = pd.concat([mlp_v1_2016_results.toPandas(),
                            mlp_v1_2017_results.toPandas(),
                            mlp_v1_2018_results.toPandas(),
                            mlp_v1_2019_results.toPandas(),
                            ])
mlp_v2_results = pd.concat([mlp_v2_2016_results.toPandas(),
                            mlp_v2_2017_results.toPandas(),
                            mlp_v2_2018_results.toPandas(),
                            mlp_v2_2019_results.toPandas(),
                            ])
rf_results

# COMMAND ----------

# Load all metrics into respective lists for plotting.
rf_accuracies = rf_results['results'].map(lambda x: x['accuracy']).values
rf_f1_scores = rf_results['results'].map(lambda x: x['f1score']).values
rf_precisions = rf_results['results'].map(lambda x: x['precision']).values
rf_recalls = rf_results['results'].map(lambda x: x['recall']).values

gbt_accuracies = gbt_results['results'].map(lambda x: x['accuracy']).values
gbt_f1_scores = gbt_results['results'].map(lambda x: x['f1score']).values
gbt_precisions = gbt_results['results'].map(lambda x: x['precision']).values
gbt_recalls = gbt_results['results'].map(lambda x: x['recall']).values

mlp_v1_accuracies = mlp_v1_results['results'].map(lambda x: x['accuracy']).values
mlp_v1_f1_scores = mlp_v1_results['results'].map(lambda x: x['f1score']).values
mlp_v1_precisions = mlp_v1_results['results'].map(lambda x: x['precision']).values
mlp_v1_recalls = mlp_v1_results['results'].map(lambda x: x['recall']).values

mlp_v2_accuracies = mlp_v2_results['results'].map(lambda x: x['accuracy']).values
mlp_v2_f1_scores = mlp_v2_results['results'].map(lambda x: x['f1score']).values
mlp_v2_precisions = mlp_v2_results['results'].map(lambda x: x['precision']).values
mlp_v2_recalls = mlp_v2_results['results'].map(lambda x: x['recall']).values

# COMMAND ----------

import matplotlib.pyplot as plt

x = list(range(2016, 2020))

plt.plot(x, rf_accuracies, label='accuracy', color='yellow')
plt.plot(x, rf_precisions, label='precision', linestyle='dashed')
plt.plot(x, rf_recalls, label='recall', linestyle='dashed', color = 'black')
plt.plot(x, rf_f1_scores, label='f1', linestyle='dashed')

plt.legend()
plt.title('Evaluation metrics for Random Forest Cross-Validation')
plt.xlabel('Evaluation Year (Training on all previous years from 2015 onward)')
plt.xticks([2016,2017,2018,2019])
plt.show()

# COMMAND ----------

x = list(range(2016, 2020))

plt.plot(x, gbt_accuracies, label='accuracy', color='yellow')
plt.plot(x, gbt_precisions, label='precision', linestyle='dashed')
plt.plot(x, gbt_recalls, label='recall', linestyle='dashed', color = 'black')
plt.plot(x, gbt_f1_scores, label='f1', linestyle='dashed')

plt.legend()
plt.title('Evaluation metrics for Gradient Boosted Trees Cross-Validation')
plt.xlabel('Evaluation Year (Training on all previous years from 2015 onward)')
plt.xticks([2016,2017,2018,2019])
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

x = list(range(2016, 2020))

plt.plot(x, mlp_v1_accuracies, label='accuracy', color='yellow')
plt.plot(x, mlp_v1_precisions, label='precision', linestyle='dashed')
plt.plot(x, mlp_v1_recalls, label='recall', linestyle='dashed', color = 'black')
plt.plot(x, mlp_v1_f1_scores, label='f1', linestyle='dashed')

plt.legend()
plt.title('Evaluation metrics for Multi-Layer Perceptron (Version 1) Cross-Validation')
plt.xlabel('Evaluation Year (Training on all previous years from 2015 onward)')
plt.xticks([2016,2017,2018,2019])
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

x = list(range(2016, 2020))

plt.plot(x, mlp_v2_accuracies, label='accuracy', color='yellow')
plt.plot(x, mlp_v2_precisions, label='precision', linestyle='dashed')
plt.plot(x, mlp_v2_recalls, label='recall', linestyle='dashed', color = 'black')
plt.plot(x, mlp_v2_f1_scores, label='f1', linestyle='dashed')

plt.legend()
plt.title('Evaluation metrics for Multi-Layer Perceptron (Version 2) Cross-Validation')
plt.xlabel('Evaluation Year (Training on all previous years from 2015 onward)')
plt.xticks([2016,2017,2018,2019])
plt.show()

# COMMAND ----------

