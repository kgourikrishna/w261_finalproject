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

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer, MinMaxScaler, OneHotEncoder, HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier, GBTClassifier
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

df_sep_1y  = df_sep_1y .withColumnRenamed("DEP_DEL15", "label")

# COMMAND ----------

display(df_sep_1y.rdd.count())

# COMMAND ----------

display(len(df_sep_1y.columns))

# COMMAND ----------

df_sep_1y.printSchema()

# COMMAND ----------

df_sep_1y = df_sep_1y.withColumn('OP_CARRIER_FL_NUM', col('OP_CARRIER_FL_NUM').cast('numeric'))

# COMMAND ----------

df_1y_test = df_sep_1y.filter(df_sep_1y.QUARTER == 4)
df_1y_train = df_sep_1y.filter(df_sep_1y.QUARTER != 4)

display(df_1y_train)

# COMMAND ----------

# Convert any remaining categorical columns to strings
df_sep_1y = df_sep_1y.withColumn('QUARTER', col('QUARTER').cast('string'))
df_sep_1y = df_sep_1y.withColumn('DAY_OF_MONTH', col('DAY_OF_MONTH').cast('string'))
df_sep_1y = df_sep_1y.withColumn('DAY_OF_WEEK', col('DAY_OF_WEEK').cast('string'))
df_sep_1y = df_sep_1y.withColumn('YEAR', col('YEAR').cast('string'))
df_sep_1y = df_sep_1y.withColumn('MONTH', col('MONTH').cast('string'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Model Stages

# COMMAND ----------


# Create categorical cols list
categorical_cols = [field for (field, dataType) in df_1y_train.dtypes if dataType == "string"]

# Drop flight number as flight number will pass through HashingTF instead of StringIndexer and OneHotCol
categorical_cols = categorical_cols.drop('OP_CARRIER_FL_NUM')

# Pass categorcial cols through STring indexer
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

# Pass categorical cols through one_hot_encoder
one_hot_vec = [x + "Vec" for x in categorical_cols]
one_hot_encode = OneHotEncoder(inputCols=index_output_cols, 
                               outputCols=one_hot_vec) 

# Pass flight number through hasher
hasher = HashingTF(inputCol='OP_CARRIER_FL_NUM', outputCol='OP_CARRIER_FL_NUM_HASH')                               


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
numeric_cols = [field for (field, dataType) in df_1y_train .dtypes if ((dataType == "double") & (field != "label"))]
imputer = Imputer(inputCols = numeric_cols, outputCols = numeric_cols)

# Assemble vector of inputs
assembler_inputs =  one_hot_vec + numeric_cols + one_hot_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="vectorized_features", handleInvalid="skip")

# Scale inputs
scaler = MinMaxScaler(inputCol="vectorized_features", outputCol="features")

# Create LogReg Model base
lg = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "features",labelCol='label')

# Create Random Forest Model base
rf = RandomForestClassifier(labelCol="label", featuresCol="vectorized_features", maxBins=100, seed=42)

# Create Gradient Boosted Decision Tree base
gbt = GBTClassifier(labelCol="label", featuresCol="vectorized_features", maxBins=100, seed=42)

# MLP v1 (1 hiden layer, 2 neurons, 1 output)
layers_v1 = [len(vec_assembler .getInputCols()), 2, 1]
mlp_v1 = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=10,
                                            layers=layers,
                                            blockSize=128,
                                            seed=1234)

# MLP v1 (2 hiden layer, 2 neurons each, 1 output)
layers_v2 = [len(vec_assembler .getInputCols()), 2,2, 1]
mlp_v2 = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=10,
                                            layers=layers,
                                            blockSize=128,
                                            seed=1234)




# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Pipelines

# COMMAND ----------

lg_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, scaler, lg])
rf_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, rf])

# Model evaluation
evaluator = MulticlassClassificationEvaluator(metricName='logLoss')

evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluatorAcc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluatorPre = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="weightedPrecision")
evaluatorRec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# COMMAND ----------

one_hot_encode.getOutputCols()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deal with Data Imbalance

# COMMAND ----------

minor_df = df_1y_train.filter(col('label') == 1)
major_df = df_1y_train.filter(col('label') == 0)
ratio = int(major_df.count()/minor_df.count())

# COMMAND ----------

ratio

# COMMAND ----------

from pyspark.sql.functions import array
#Sample the target variable and fix the issue of the imbalanced 
a = range(ratio)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", F.explode(array([F.lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
train_df_oversampled = major_df.unionAll(oversampled_df)
display(train_df_oversampled.groupby('label').count())

# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Random Forest

# COMMAND ----------

### Testing basic RF model

# Fitting the model on training data 
rf_fit_model = rf_pipeline.fit(df_1y_train) 
  
# Storing the results on test data 
rf_results = rf_fit_model.transform(df_1y_test) 

# Evaluate the model
f1score = evaluatorF1.evaluate(rf_results)
accuracy = evaluatorAcc.evaluate(rf_results)
precision = evaluatorPre.evaluate(rf_results)
recall = evaluatorRec.evaluate(rf_results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

display(rf_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning - Random Forest

# COMMAND ----------

from hyperopt import fmin, tpe, hp, STATUS_OK

#train_df, val_df= train_df_oversampled.randomSplit([.9, .1], seed=42)
#train_df, val_df= df_1y_train.randomSplit([.9, .1], seed=42)

#train_df = df_sep_1y.filter(df_sep_1y.MONTH != 9)
#val_df = df_sep_1y.filter(df_sep_1y.MONTH == 9)

train_df = train_df_oversampled.filter(df_sep_1y.MONTH != 9)
val_df = train_df_oversampled.filter(df_sep_1y.MONTH == 9)

train_df.cache()
val_df.cache()
def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    estimator = rf_pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
    model = estimator.fit(train_df)
    train = model.transform(train_df)
    train_accuracy = evaluatorAcc.evaluate(train)

    # Evaluation
    preds = model.transform(val_df)
    log_loss = evaluator.evaluate(preds)
    val_accuracy = evaluatorAcc.evaluate(preds)

    return {
        'loss': log_loss,
        'status': STATUS_OK,
        # -- store other results like this
        'Accuracies': {'Train': train_accuracy, 'Validation': val_accuracy},
        }

# COMMAND ----------

from hyperopt import hp

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "num_trees": hp.quniform("num_trees", 10, 100, 1)
}

# COMMAND ----------

mlflow.pyspark.ml.autolog(log_models=False)

num_evals = 10
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))
best_hyperparam



# COMMAND ----------

trials.trials

# COMMAND ----------

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
    estimator = rf_pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})
    combined_df = train_df.union(val_df) # Combine train & validation together

    model = estimator.fit(combined_df)
    preds = model.transform(df_1y_test)

    # Evaluate
    log_loss = evaluator.evaluate(preds)
    f2_evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=2)
    f2_score = f2_evaluator.evaluate(preds)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.log_metric("f2_score", f2_score)
    mlflow.spark.log_model(model, "Random-Forest-model")

# COMMAND ----------

import mlflow
logged_model = 'runs:/b8cdc03baf904aab8a11fabf5a4d1572/Random-Forest-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_1y_test)

# COMMAND ----------

# Evaluate the model
f1score = evaluatorF1.evaluate(preds)
accuracy = evaluatorAcc.evaluate(preds)
precision = evaluatorPre.evaluate(preds)
recall = evaluatorRec.evaluate(preds)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col

def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())

# COMMAND ----------

display(preds)

# COMMAND ----------


#result_df = preds.filter((preds["prediction"] == 0) & (preds["Label"] == 1))

df2 = preds.withColumn("prob_flag", extract_prob_udf(col("probability"))).toPandas()

# COMMAND ----------

df2.filter(df2['prediction'] == 1).count()

# COMMAND ----------

import matplotlib.pyplot as plt

positives = df2[df2['prediction'] == 1]
negatives = df2[df2['prediction'] == 0]

plt.hist(positives['prob_flag'], bins=30, color='orange', edgecolor='black', alpha = .5, label = '1')
plt.hist(negatives['prob_flag'], bins=30, color='skyblue', edgecolor='black', alpha = .5, label = '0')


# Add labels and title
plt.xlabel('probabilities for delay')
plt.ylabel('Frequency')
plt.title('Histogram of probabilities for Flight Delay: RF Model')
plt.legend()

# Show plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Logistic Regression

# COMMAND ----------

### Testing basic LG model

# Fitting the model on training data 
fit_model = lg_pipeline.fit(df_1y_train) 
  
# Storing the results on test data 
results = fit_model.transform(df_1y_test) 

display(results)

# COMMAND ----------

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

display(results.groupby('prediction').count())

# COMMAND ----------

fit_model.summary