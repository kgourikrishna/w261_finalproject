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

display(dbutils.fs.ls(f"{team_blob_url}/pageranks"))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer, MinMaxScaler, OneHotEncoder, PCA
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

df_sep_1y = spark.read.parquet(f"{team_blob_url}/1year_cleaned_new")
df_60m = spark.read.parquet(f"{team_blob_url}/60month_cleaned_new_1")


# COMMAND ----------

df_60m.count()

# COMMAND ----------

pr_2018 = spark.read.parquet(f"{team_blob_url}/pageranks/_2018")

# COMMAND ----------

df_train = df_60m.filter(df_60m.YEAR != 2019)
df_test_2019 = df_60m.filter(df_60m.YEAR == 2019)

# COMMAND ----------


df_train = df_train.join(pr_2018, df_train.ORIGIN_AIRPORT_ID == pr_2018.id, how='left')
df_test_2019 = df_test_2019.join(pr_2018, df_test_2019.ORIGIN_AIRPORT_ID == pr_2018.id, how='left')

# COMMAND ----------

#Drop additional unused columns 
#df_sep_1y = df_sep_1y.drop('OP_CARRIER_AIRLINE_ID','ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','YEAR','OP_CARRIER_FL_NUM')
df_train = df_train.drop('OP_CARRIER_AIRLINE_ID','ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','OP_CARRIER_FL_NUM','id','DISTANCE_GROUP','QUARTER')
df_test_2019 = df_test_2019.drop('OP_CARRIER_AIRLINE_ID','ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','OP_CARRIER_FL_NUM','id','DISTANCE_GROUP','QUARTER')

# COMMAND ----------

#df_sep_1y  = df_sep_1y.withColumnRenamed("DEP_DEL15", "label")
#df_sep_1y  = df_sep_1y.withColumn('label', col('label').cast('int'))

df_train  = df_train.withColumnRenamed("DEP_DEL15", "label")
df_train  = df_train.withColumn('label', col('label').cast('int'))

df_test_2019  = df_test_2019 .withColumnRenamed("DEP_DEL15", "label")
df_test_2019  = df_test_2019 .withColumn('label', col('label').cast('int'))

# COMMAND ----------

# Convert any remaining categorical columns to strings
#df_sep_1y = df_sep_1y.withColumn('DAY_OF_MONTH', col('DAY_OF_MONTH').cast('string'))
#df_sep_1y = df_sep_1y.withColumn('DAY_OF_WEEK', col('DAY_OF_WEEK').cast('string'))

df_train = df_train.withColumn('DAY_OF_MONTH', col('DAY_OF_MONTH').cast('string'))
df_train = df_train.withColumn('DAY_OF_WEEK', col('DAY_OF_WEEK').cast('string'))
df_train = df_train.withColumn('F_MONTH', col('F_MONTH').cast('string'))


df_test_2019 = df_test_2019 .withColumn('DAY_OF_MONTH', col('DAY_OF_MONTH').cast('string'))
df_test_2019  = df_test_2019 .withColumn('DAY_OF_WEEK', col('DAY_OF_WEEK').cast('string'))
df_test_2019  = df_test_2019 .withColumn('F_MONTH', col('F_MONTH').cast('string'))

# COMMAND ----------

#df_1y_test = df_1y_test.drop('QUARTER')
#df_1y_train = df_1y_test.drop('QUARTER')

df_train = df_train.drop('YEAR')
df_test_2019 = df_test_2019.drop('YEAR')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Model Stages

# COMMAND ----------


# Create categorical cols list
categorical_cols = [field for (field, dataType) in df_train.dtypes if dataType == "string"]

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
numeric_cols = [field for (field, dataType) in df_train .dtypes if ((dataType == "double") & (field != "label"))]
imputer = Imputer(inputCols = numeric_cols, outputCols = numeric_cols)

# Assemble vector of inputs
assembler_inputs =  one_hot_vec + numeric_cols + one_hot_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="vectorized_features", handleInvalid="skip")

# Scale inputs
scaler = MinMaxScaler(inputCol="vectorized_features", outputCol="features")

# PCA for MLP Models
pca = PCA(k=12,inputCol='features', outputCol='pcaFeature')

# Create Random Forest Model base
rf = RandomForestClassifier(labelCol="label", featuresCol="vectorized_features", maxBins=40, seed=42)

# Create Gradient Boosted Decision Tree base
gbt = GBTClassifier(labelCol="label", featuresCol="vectorized_features", maxBins=40, seed=42)

# MLP v1 (1 hidden layer, 5 neurons, 1 output)
layers_v1 = [884, 5, 2]
mlp_v1 = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=100,
                                            layers=layers_v1)

# MLP v1 (2 hidden layers, 10 neurons each, 1 output)
layers_v2 = [884,10,10, 2]
mlp_v2 = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='features',
                                            maxIter=100,
                                            layers=layers_v2
                                            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Pipelines

# COMMAND ----------

# Random Forest and GBT Pipelines
rf_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, rf])

# MLP Pipelines
mlp_v1_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, scaler, mlp_v1])
mlp_v2_pipeline = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, scaler, mlp_v2])

# Model Evaluation Metrics
evaluator = MulticlassClassificationEvaluator(metricName='logLoss')
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluatorAcc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluatorPre = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="weightedPrecision")
evaluatorRec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Deal with Data Imbalance

# COMMAND ----------

minor_df = df_train.filter(col('label') == 1)
major_df = df_train.filter(col('label') == 0)
ratio = int(major_df.count()/minor_df.count())

# COMMAND ----------

from pyspark.sql.functions import array
#Sample the target variable and fix the issue of the imbalanced 
a = range(ratio)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", F.explode(array([F.lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
train_df_oversampled = major_df.unionAll(oversampled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Random Forest

# COMMAND ----------

### Testing basic RF model

# Fitting the model on training data 
rf_fit_model = rf_pipeline.fit(df_train) 
  
# Storing the results on test data 
rf_results = rf_fit_model.transform(df_test_2019) 

display(rf_results)

# Evaluate the model
f1score = evaluatorF1.evaluate(rf_results)
accuracy = evaluatorAcc.evaluate(rf_results)
precision = evaluatorPre.evaluate(rf_results)
recall = evaluatorRec.evaluate(rf_results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Gradient Boosted Trees Model

# COMMAND ----------

### Testing basic GBT model

# Fitting the model on training data 
gbt_fit_model = gbt_pipeline.fit(df_train)

  
# Storing the results on test data 
gbt_results = gbt_fit_model.transform(df_test_2019) 

display(gbt_results)

# Evaluate the model
f1score = evaluatorF1.evaluate(gbt_results)
accuracy = evaluatorAcc.evaluate(gbt_results)
precision = evaluatorPre.evaluate(gbt_results)
recall = evaluatorRec.evaluate(gbt_results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Tuning

# COMMAND ----------

from hyperopt import fmin, tpe, hp, STATUS_OK

train_df, val_df= df_train.randomSplit([.9, .1], seed=42)
#train_df, val_df= train_df_oversampled_60.randomSplit([.9, .1], seed=42)

train_df.cache()
val_df.cache()

def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]
    num_bins = params["num_bins"]

    estimator = rf_pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees, rf.maxBins:num_bins})
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
    "num_trees": hp.quniform("num_trees", 10, 100, 1),
    "num_bins": hp.quniform("num_bins", 2, 64, 1)
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
                       rstate=np.random.default_rng(42)
                       )
best_hyperparam

# COMMAND ----------

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
    best_num_bins = best_hyperparam["num_bins"]
    
    estimator = rf_pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees, rf.maxBins: best_num_bins})
    
    combined_df = train_df.union(val_df) # Combine train & validation together

    model = estimator.fit(combined_df)
    preds = model.transform(df_test_2019)

    # Evaluate
    log_loss = evaluator.evaluate(preds)
    f2_evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=2)
    f2_score = f2_evaluator.evaluate(preds)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_param("numbins", best_num_bins)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.log_metric("f2_score", f2_score)
    mlflow.spark.log_model(model, "Random-Forest-model")

# COMMAND ----------

import mlflow
logged_model = 'runs:/9691abb766bb4db1b1a3fc784c8ef6df/Random-Forest-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_sep_60m_test)

# Evaluate the model
f1score = evaluatorF1.evaluate(preds)
accuracy = evaluatorAcc.evaluate(preds)
precision = evaluatorPre.evaluate(preds)
recall = evaluatorRec.evaluate(preds)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gradient Boosted Trees Tuning

# COMMAND ----------

train_df, val_df= df_train.randomSplit([.9, .1], seed=42)
#train_df, val_df= train_df_oversampled_60.randomSplit([.9, .1], seed=42)

train_df.cache()
val_df.cache()

def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    max_bins = params["max_bins"]
    min_inst = params["minInstancesPerNode"]



    estimator = gbt_pipeline.copy({gbt.maxDepth: max_depth, gbt.maxBins: max_bins, gbt.minInstancesPerNode:min_inst})
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

# COMMAND ----------

from hyperopt import hp

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "max_bins": hp.quniform("max_bins", 2, 64, 1),
    "minInstancesPerNode": hp.quniform("minInstancesPerNode", 1, 10, 1)
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
                       rstate=np.random.default_rng(42),
                       return_argmin=False
                       )
best_hyperparam

# COMMAND ----------

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_bins = best_hyperparam["max_bins"]
    best_inst = best_hyperparam["minInstancesPerNode"]
    #best_ss = best_hyperparam["stepSize"]
    #best_subs = best_hyperparam["subsamplingRate"]
    


    estimator = gbt_pipeline.copy({gbt.maxDepth: best_max_depth, gbt.maxBins: best_num_bins, gbt.minInstancesPerNode:best_inst})
    combined_df = train_df.union(val_df) # Combine train & validation together

    model = estimator.fit(combined_df)
    preds = model.transform(df_test_2019)

    # Evaluate
    log_loss = evaluator.evaluate(preds)
    f2_evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=2)
    f2_score = f2_evaluator.evaluate(preds)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numBins", best_num_bins)
    mlflow.log_param("numInst_per_node", best_inst)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.log_metric("f2_score", f2_score)
    mlflow.spark.log_model(model, "Gradient-Boost-Trees-model-60")

# COMMAND ----------

## Look at Model Feature Importance

coefs = list(zip(assembler_inputs, model.stages[-1].featureImportances))
coefs.sort(key=lambda tup: abs(tup[1]))
coefs

# COMMAND ----------

import mlflow
logged_model = 'runs:/08d74f3f3eb34a8abb01f42ba261ee94/Gradient-Boost-Trees-model-60'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_test_2019)

# Evaluate the model
f1score = evaluatorF1.evaluate(preds)
accuracy = evaluatorAcc.evaluate(preds)
precision = evaluatorPre.evaluate(preds)
recall = evaluatorRec.evaluate(preds)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLP Version 1

# COMMAND ----------

### Testing Version 1 MLP model

# Fitting the model on training data 
mlp1_fit_model = mlp_v1_pipeline.fit(df_train) 
  
# Storing the results on test data 
mlp_results = mlp1_fit_model.transform(df_test_2019) 

display(mlp_results)

# Evaluate the model
f1score = evaluatorF1.evaluate(mlp_results)
accuracy = evaluatorAcc.evaluate(mlp_results)
precision = evaluatorPre.evaluate(mlp_results)
recall = evaluatorRec.evaluate(mlp_results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

### MLP 1 After Oversampling

# Fitting the model on training data 
mlp1_fit_model = mlp_v1_pipeline.fit(train_df_oversampled) 
  
# Storing the results on test data 
mlp_results2 = mlp1_fit_model.transform(df_test_2019) 

display(mlp_results2)

# Evaluate the model
f1score = evaluatorF1.evaluate(mlp_results2)
accuracy = evaluatorAcc.evaluate(mlp_results2)
precision = evaluatorPre.evaluate(mlp_results2)
recall = evaluatorRec.evaluate(mlp_results2)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

#Hyperparameter tuning on MLP V1
from hyperopt import fmin, tpe, hp, STATUS_OK

#train_df, val_df= train_df_oversampled.randomSplit([.9, .1], seed=42)
train_df, val_df= df_train.randomSplit([.9, .1], seed=42)

train_df.cache()
val_df.cache()
def objective_function(params):    
    # set the hyperparameters that we want to tune
    solver = params["solver"]
    stepSize = params["stepSize"]

    estimator = mlp_v1_pipeline.copy({mlp_v1.solver: solver, mlp_v1.stepSize: stepSize})
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
    "solver": hp.choice("solver", ['gd','l-bfgs']),
    "stepSize": hp.quniform("stepSize", 0.01, 0.5, 0.01)
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
                       return_argmin=False,
                       rstate=np.random.default_rng(42))
best_hyperparam

# COMMAND ----------

best_hyperparam

# COMMAND ----------

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_solver = best_hyperparam['solver']
    best_stepSize = best_hyperparam["stepSize"]
    estimator = mlp_v1_pipeline.copy({mlp_v1.solver: best_solver, mlp_v1.stepSize: best_stepSize})
    combined_df = train_df.union(val_df) # Combine train & validation together

    model = estimator.fit(df_train)
    preds = model.transform(df_test_2019)

    # Evaluate
    log_loss = evaluator.evaluate(preds)
    f2_evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=2)
    f2_score = f2_evaluator.evaluate(preds)

    # Log param and metrics for the final model
    mlflow.log_param("solver", best_solver)
    mlflow.log_param("stepSize", best_stepSize)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.log_metric("f2_score", f2_score)
    mlflow.spark.log_model(model, "MLP-V1-model")

# COMMAND ----------

import mlflow
logged_model = 'runs:/cb49d3e9dce5493e82a4751d9dbc22d4/MLP-V1-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_test_2019)

# Evaluate the model
f1score = evaluatorF1.evaluate(preds)
accuracy = evaluatorAcc.evaluate(preds)
precision = evaluatorPre.evaluate(preds)
recall = evaluatorRec.evaluate(preds)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLP Version 2

# COMMAND ----------

### Testing Version 2 MLP model

# Fitting the model on training data 
mlp2_fit_model = mlp_v2_pipeline.fit(df_train) 
  
# Storing the results on test data 
results = mlp2_fit_model.transform(df_test_2019) 

display(results)

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

### Testing Version 2 MLP model

# Fitting the model on training data 
mlp2_fit_model = mlp_v2_pipeline.fit(train_df_oversampled) 
  
# Storing the results on test data 
results = mlp2_fit_model.transform(df_test_2019) 

display(results)

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLP Version 2 Tuning

# COMMAND ----------

#Hyperparameter tuning on MLP V1
from hyperopt import fmin, tpe, hp, STATUS_OK

#train_df, val_df= train_df_oversampled.randomSplit([.9, .1], seed=42)
train_df, val_df= df_train.randomSplit([.9, .1], seed=42)

train_df.cache()
val_df.cache()
def objective_function(params):    
    # set the hyperparameters that we want to tune
    solver = params["solver"]
    stepSize = params["stepSize"]

    estimator = mlp_v2_pipeline.copy({mlp_v2.solver: solver, mlp_v2.stepSize: stepSize})
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
    "solver": hp.choice("solver", ['gd','l-bfgs']),
    "stepSize": hp.quniform("stepSize", 0.01, 0.5, 0.01)
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
                       return_argmin=False,
                       rstate=np.random.default_rng(42))
best_hyperparam

# COMMAND ----------

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_solver = best_hyperparam['solver']
    best_stepSize = best_hyperparam["stepSize"]
    estimator = mlp_v2_pipeline.copy({mlp_v2.solver: best_solver, mlp_v2.stepSize: best_stepSize})
    combined_df = train_df.union(val_df) # Combine train & validation together

    model = estimator.fit(df_train)
    preds = model.transform(df_test_2019)

    # Evaluate
    log_loss = evaluator.evaluate(preds)
    f2_evaluator = MulticlassClassificationEvaluator(metricName='fMeasureByLabel', beta=2)
    f2_score = f2_evaluator.evaluate(preds)

    # Log param and metrics for the final model
    mlflow.log_param("solver", best_solver)
    mlflow.log_param("stepSize", best_stepSize)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.log_metric("f2_score", f2_score)
    mlflow.spark.log_model(model, "MLP-V2-model")

# COMMAND ----------

import mlflow
logged_model = 'runs:/1518e422c260410493e5beab8c6171c9/MLP-V2-model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_test_2019)

# Evaluate the model
f1score = evaluatorF1.evaluate(preds)
accuracy = evaluatorAcc.evaluate(preds)
precision = evaluatorPre.evaluate(preds)
recall = evaluatorRec.evaluate(preds)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run MLP Version 1 with PCA

# COMMAND ----------

### Testing Version 1 MLP model with PCA

# Create new model with new input features
layers_pca_v1 = [12,5,2]
mlp_v1_pca = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='pcaFeature',
                                            maxIter=100,
                                            layers=layers_pca_v1 
                                            )

# New PCA pipeline
mlp_v1_pipeline_pca = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, scaler,pca, mlp_v1_pca])

# Fitting the model on training data 
mlp1_pca_fit_model = mlp_v1_pipeline_pca.fit(df_train) 
  
# Storing the results on test data 
results = mlp1_pca_fit_model.transform(df_test_2019) 

display(results)

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# Fitting the model on training data (oversampled)
mlp1_pca_fit_model = mlp_v1_pipeline_pca.fit(train_df_oversampled) 
  
# Storing the results on test data 
results = mlp1_pca_fit_model.transform(df_test_2019) 

display(results)

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run MLP Version 2 with PCA

# COMMAND ----------

### Testing Version 2 MLP model with PCA

# Create new model with new input features
layers_pca_v2 = [12,10,10,2]
mlp_v2_pca = MultilayerPerceptronClassifier(labelCol='label',
                                            featuresCol='pcaFeature',
                                            maxIter=100,
                                            layers=layers_pca_v2 
                                            )

# New PCA pipeline
mlp_v2_pipeline_pca = Pipeline(stages=[string_indexer,one_hot_encode,imputer, vec_assembler, scaler,pca, mlp_v2_pca])

# Fitting the model on training data 
mlp2_pca_fit_model = mlp_v2_pipeline_pca.fit(df_train) 
  
# Storing the results on test data 
results = mlp2_pca_fit_model.transform(df_test_2019) 

display(results)

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# Fitting the model on training data 
mlp2_pca_fit_model = mlp_v2_pipeline_pca.fit(train_df_oversampled) 
  
# Storing the results on test data 
results = mlp2_pca_fit_model.transform(df_test_2019) 

display(results)

# Evaluate the model
f1score = evaluatorF1.evaluate(results)
accuracy = evaluatorAcc.evaluate(results)
precision = evaluatorPre.evaluate(results)
recall = evaluatorRec.evaluate(results)

print(f1score, accuracy, precision, recall)

# COMMAND ----------

# MAGIC %md
# MAGIC With the numerous one-hot-encoded variables, PCA may not be appropriate and capturing the necessary information. Limitation in this approach

# COMMAND ----------



# COMMAND ----------

##

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gaps Analysis

# COMMAND ----------

## Load Best Model 
import mlflow
logged_model = 'runs:/08d74f3f3eb34a8abb01f42ba261ee94/Gradient-Boost-Trees-model-60'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
preds = loaded_model.transform(df_test_2019)

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

df2 = preds.withColumn("prob_score", extract_prob_udf(col("probability"))).toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

positives = df2[df2['prediction'] == 1]
negatives = df2[df2['prediction'] == 0]

plt.hist(positives['prob_score'], bins=30, color='orange', edgecolor='black', alpha = .5, label = '1')
plt.hist(negatives['prob_score'], bins=30, color='skyblue', edgecolor='black', alpha = .5, label = '0')


# Add labels and title
plt.xlabel('probabilities for delay')
plt.ylabel('Frequency')
plt.title('Histogram of probabilities for Flight Delay: GBT Model')
plt.legend()

# Show plot
plt.show()