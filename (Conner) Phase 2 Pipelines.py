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
secret_scope   = "team_2_1"            # The name of the scope created in your local computer using the Databricks CLI
secret_key  = "team_2_1_key"             # The name of the secret key created in your local computer using the Databricks CLI
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

team_blob_url 

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


# COMMAND ----------

df_sep_1y = df_sep_1y.drop(
 'OP_CARRIER_AIRLINE_ID',
 'ORIGIN_AIRPORT_ID',
 'DEST_AIRPORT_ID')

# COMMAND ----------

df_sep_1y = df_sep_1y.withColumn('OP_CARRIER_FL_NUM', col('OP_CARRIER_FL_NUM').cast('double'))

# COMMAND ----------

df_sep_1y  = df_sep_1y .withColumnRenamed("DEP_DEL15", "label")

# COMMAND ----------

df_1y_test = df_sep_1y.filter(df_sep_1y.QUARTER == 4)
df_1y_train = df_sep_1y.filter(df_sep_1y.QUARTER != 4)

display(df_1y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Model Stages

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Logistic Regression Pipeline

# COMMAND ----------



# COMMAND ----------




# COMMAND ----------


categorical_cols = [field for (field, dataType) in df_1y_train. dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

one_hot_vec = [x + "Vec" for x in categorical_cols]

one_hot_encode = OneHotEncoder(inputCols=index_output_cols, 
                               outputCols=one_hot_vec) 

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

numeric_cols = [field for (field, dataType) in df_1y_train.dtypes if ((dataType == "double") & (field != "label"))]

imputer = Imputer(inputCols = numeric_cols, outputCols = numeric_cols)

assembler_inputs =  one_hot_vec + numeric_cols + one_hot_cols

vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="vectorized_features", handleInvalid="skip")

scaler = MinMaxScaler(inputCol="vectorized_features", outputCol="features")

lg = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "features",labelCol='label')


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

lg_pipeline = Pipeline(stages=[string_indexer, one_hot_encode, imputer, vec_assembler, scaler, lg])


# Model evaluation
evaluator = MulticlassClassificationEvaluator(metricName='logLoss')

evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluatorAcc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluatorPre = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="weightedPrecision")
evaluatorRec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Deal With Imbalanced Data

# COMMAND ----------



# COMMAND ----------

minor_df = df_1y_train.filter(col('label') == 1)
major_df = df_1y_train.filter(col('label') == 0)
ratio = int(major_df.count()/minor_df.count())

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
# MAGIC ##Train and Test Baseline Model 

# COMMAND ----------

lg_fit_model = lg_pipeline.fit(train_df_oversampled)

lg_results = lg_fit_model.transform(df_1y_test) 

# Evaluate the model
f1score = evaluatorF1.evaluate(lg_results)
accuracy = evaluatorAcc.evaluate(lg_results)
precision = evaluatorPre.evaluate(lg_results)
recall = evaluatorRec.evaluate(lg_results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

print(f1score, accuracy, precision, recall)

# COMMAND ----------

display(lg_results.groupby('prediction').count())

# COMMAND ----------



# COMMAND ----------

##storing coefs of lg model 
coefs = list(zip(assembler_inputs, lg_fit_model.stages[-1].coefficients))
coefs.sort(key=lambda tup: abs(tup[1]))



# COMMAND ----------

# MAGIC %md
# MAGIC ##HyperTuning Logistic Regression 

# COMMAND ----------

from hyperopt import fmin, tpe, hp, STATUS_OK

train_df, val_df= train_df_oversampled.randomSplit([.9, .1], seed=42)
train_df.cache()
val_df.cache()

def objective_function(params):    
    # set the hyperparameters that we want to tune
    penalty = params['elasticNetParam']
    regParam = params['regParam']
    estimator = lg_pipeline.copy({lg.elasticNetParam: penalty, lg.regParam: regParam})
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

from hyperopt import fmin, tpe, hp, STATUS_OK

#train_df, val_df= train_df_oversampled.randomSplit([.9, .1], seed=42)
train_df, val_df= df_1y_train.randomSplit([.9, .1], seed=42)
train_df.cache()
val_df.cache()


search_space = {
'elasticNetParam': hp.uniform('elasticNetParam', 0, 1.0),
'regParam': hp.uniform('regParam', 0, 1.0)}

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

with mlflow.start_run():
    best_penalty = best_hyperparam['elasticNetParam']
    best_regParam = best_hyperparam['regParam']
    estimator = lg_pipeline.copy({lg.elasticNetParam: best_penalty, lg.regParam: best_regParam})
    combined_df = train_df.union(val_df) # Combine train & validation together

    model = estimator.fit(combined_df)
    preds = model.transform(df_1y_test)

    # Evaluate
    log_loss = evaluator.evaluate(preds)
    f2_evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=2)
    f2_score = f2_evaluator.evaluate(preds)

    # Log param and metrics for the final model
    mlflow.log_param("elasticNetParam", best_penalty)
    mlflow.log_param("regParam", best_regParam)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.log_metric("f2_score", f2_score)
    mlflow.spark.log_model(model, "Logistic-Regression-model")

# COMMAND ----------

f2_score

# COMMAND ----------

model.stages[-1].trees

# COMMAND ----------

coefs = list(zip(assembler_inputs, model.stages[-1].coefficients))
coefs.sort(key=lambda tup: abs(tup[1]))
coefs

# COMMAND ----------

logged_model = 'runs:/91dbd0c2f88843cebdad3dcdee223f8c/Logistic-Regression-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_1y_test)
display(preds)

# COMMAND ----------

display(preds.groupby('prediction').count())

# COMMAND ----------

f1score = evaluatorF1.evaluate(preds)
accuracy = evaluatorAcc.evaluate(preds)
precision = evaluatorPre.evaluate(preds)
recall = evaluatorRec.evaluate(preds)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Scala version implements .roc() and .pr()
# Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
# Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

import matplotlib.pyplot as plt

preds1 = preds.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))

# Returns as a list (false positive rate, true positive rate)
points = CurveMetrics(preds1).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x_val, y_val)

# COMMAND ----------

import numpy as np

thresholds = np.linspace(0, 1, 100)
f1_scores = []

for threshold in thresholds:
    loaded_model.stages[-1].setThreshold(threshold)
    preds = loaded_model.transform(df_1y_test)
    f1score = evaluatorF1.evaluate(preds)
    f1_scores.append(f1score)



# COMMAND ----------

plt.figure()
plt.title('Threshold vs. F1')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.plot(thresholds , f1_scores)

