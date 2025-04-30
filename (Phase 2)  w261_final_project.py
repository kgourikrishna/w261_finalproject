# Databricks notebook source
# MAGIC %md
# MAGIC # Up in the Air: Predicting Flight Delays

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Jason Frankel (jasonfrankel@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Jason%20Headshot.png?raw=true'>
# MAGIC
# MAGIC Rodrigo Cochran (rodrigocochran@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Rodrigo%20Headshot.png?raw=true'>
# MAGIC
# MAGIC Conner Davis (connerdavis@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Conner%20Headshot.png?raw=true'>
# MAGIC
# MAGIC Kushal Gourikrishna (kgourikrishna@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Kushal%20Headshot.png?raw=true'>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##Phase Leader Plan:
# MAGIC | Project Week | Phase Leader |
# MAGIC | ----------- | ----------- |
# MAGIC | Week 1      | Rodrigo      |
# MAGIC | Week 2      | **Jason**        |
# MAGIC | Week 3      | **Kushal**       |
# MAGIC | Week 4      | Conner        |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Updated Credit Assignment Plan:
# MAGIC #### Phase 2
# MAGIC
# MAGIC | Project Task| Owner |Task Complete? |
# MAGIC | ----------- | ----------- |----------- |
# MAGIC | EDA on flight portion of data. Statistical tests on potential model features. Setup and run Walk-Forward Cross Validation. | Rodrigo      |Yes|
# MAGIC | EDA on weather portion of data. Setup input data for model and setup ML pipelines. Hyperparameter tuning on Random Forest Models.    | Kushal       |Yes|
# MAGIC | EDA on flight portion of data. Overall Feature engineering (correlations, etc.). Create time-based feature. Lead presentation creation | Jason      |Yes|
# MAGIC | EDA on weather portion of data. Feature engineering with weather categorical data. Hyperparameter tuning on Logistic Regression   | Conner        |Yes|

# COMMAND ----------

# MAGIC %md
# MAGIC #### Phase 3
# MAGIC
# MAGIC |Project Task|Time Estimate|Jason|Kushal|Rodrigo |Conner |
# MAGIC |:----|:----|:----|:----|:----|:----|
# MAGIC |Train more sophisticated models - Gradient Boosted Decision Tree|3 Hr||X| | |
# MAGIC |Train more sophisticated models - Multilayer Perceptron Neural Network Version 1|3 Hr| ||X| |
# MAGIC |Train more sophisticated models - Multilayer Perceptron Neural Network Version 2|3 Hr| | ||X|
# MAGIC |Train more sophisticated models - Extra Credit|3-6 Hr|X|X|X|X|
# MAGIC |Fine tune pipeline with grid search + describe performance|2 Hr| |X|X|X|
# MAGIC |Test model on 2019 data and discuss performance|1 Hr| |X|X|X|
# MAGIC |Feature Engineering on new features|2 Hr|X|| | |
# MAGIC |New Directions - Extra Credit|1 Hr|X| | | |
# MAGIC |Clean up Code|0.25 Hr|X| | | |
# MAGIC |Final Presentation|2-3 Hr|X|X | X| X|
# MAGIC |Final Report|4-6 Hr|X| X| X| X|
# MAGIC |Clean up Notebook + Submit|0.25 Hr| || |X|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project Timeline
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Gantt%20Chart%20Update.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC
# MAGIC This project is looking to examine existing data on domestic flights, time-series weather data, and weather station data to identify the primary causes of flight delays and ultimately predict flight delays. We first explored joined datasets and performed feature engineering to identify variables we would like to use for our pipelines. We then built Logistic Regression and Random Forest models with this data to evaluate the output of these pipelines against historical data to develop reliable predictions. We also performed walk forward validation on our models to assess effectiveness over time. Our results for our Logistic Regression model was an accuracy of .82 after 12 months of training data, and .792 for the Random Forest model, though these models were hyperopt tuned to values of .839 and .846 respectively. We will further tune our model to improve accurracy and reduce false positives as well as experiment with new models such as MLP Neural Networks and ensemble models.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Description
# MAGIC
# MAGIC The datasets we are working with here include 3 raw datasets, and then a pre-joined dataset. 
# MAGIC
# MAGIC The first raw dataset is domestic flight data from the bureau of transportation statistics (BTS). The data is already pre-divided into three sections: 1st quarter of 2015, first two quarters of 2015, and all of 2019 data. As one can see in the EDA below, it contains fields such as airline, flight number, departure, summary of the flight, and arrival information, including (for our purposes) whether there was a delay and what was the cause of said delay.
# MAGIC
# MAGIC The second raw dataset is global weather data from the National Oceanic and Atmospheric Administration (NOAA) repository with hourly, daily summaries, and monthly summaries of weather information from 2015-2021. This contains the locations of the weather stations used in the data as well as the time, with weather noted including temperature, wind, precipitation type + amount, air pressure, and sun rise + set to name a few. See EDA below for more information.
# MAGIC
# MAGIC The last raw dataset is more information about all of the weather stations used in the weather dataset above. Specifically it can be useful because it has all of the information of all of the other weather stations in relation to the main station in question, and includes the distance from the weather station to its neighbor. This will be useful to see if weather conditions nearby, at the arrival destination, or elsewhere in the country affect the delay of the flight at one location.
# MAGIC
# MAGIC Last is the combination of all of these datasets to get the On Time Performance and Weather (OPTW) joined dataset. OPTW comes in three versions (OPTW for Q1 2015, H1 2015, and 2019) and combines flight, weather, and station data to produce a unique record for each flight (date + tail num + airline) showing all the flight info from the flight data, with all of the available weather information at the closest station to the departure flight four hours before the flight departure time. An exploratory question would be to see if supplemental data had additional predictive power (weather at arrival station, other stations en route, etc.).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## ETL Workflow Overview
# MAGIC
# MAGIC The overall workflow was a generalized ETL process that extracted the raw data from large-file storage in the Azure blob and transformed it into a dataset that will contain the features we looked at in the first round of modeling. Transformations will included datatype cleaning, null-value analysis, categorical variable encoding, numerical variable scaling, and imputations for variables that are missing less than 10% of the data. The data was then piped into various model experiements explained later in this report. Then, for each model, the classification model evaluation metrics were calculated: accuracy, precision, recall, and F-score. During this process, we also optimized hyperparameters through Bayesian Optimization.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/ETL%20Workflow.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA and Feature Engineering
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC For flight data, we first identified a UID for flights, which is a combination of the flight_num, carrier_id, and date. We then created a couple time-based variable using window functions, first to determine whether the plane was in the air 2 hours before the given plane and if the plane in the air was delayed (see below for variables).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Time-based Variables
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Time%20Variable%20Histogram.png?raw=true'>
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20Rate%20by%20Time%20Variable.png?raw=true'>
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Flight%20Delayed%20Histogram.png?raw=true'>
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20Rate%20by%20Delay%20Variable.png?raw=true'>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Then we removed all variables we found redundant from correlational analysis, removed all arrival based information, as we cannot know that info two hours before the flight, removed all rows with a cancelled flight, non-numeric fields, and daily + monthly weather features as they were deemed uneccessary with hourly weather information currently available and more useful. We also deleted all rows with a cancelled flight since they are excluded from our analysis.
# MAGIC
# MAGIC Then we created 43 boolean variables for key weather codes and 8 boolean variables for sky codes so we could include them into the model.

# COMMAND ----------

# MAGIC %md
# MAGIC Last, before one-hot encoding variables, we ran a correlation analysis, described the remaining variables, analyzed frequency, did a pairwise input-out analysis of the variables, and included them into a data dictionary, all of which you can see below. 
# MAGIC
# MAGIC Some highlights of what we found was that there was a significant difference between carriers, airports, precipitation amounts in being able to have a significant movement with delays.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Histogram of Subset of Variables Side-by-Side
# MAGIC
# MAGIC Below is a subset of the overall input variable set and their histograms.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Histograms%20Codensed%20Cropped.jpeg?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Delay Rates for Subset of Categorical Variables Side-by-Side
# MAGIC
# MAGIC Below is a subset of the categorical varaibles and their associated delay rates. Due to the high number of categories for certain variables, only a subset is shown to preserve readability.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20Rate%20by%20Variable%20Condensed.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Correlation Matrix

# COMMAND ----------

#### Correlation Matrix
# <!-- 
# <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Correlation%20Matrix.png?raw=true'> -->

import pandas as pd
corr_matrix = pd.read_csv('correlation_matrix_eda.csv')
display(corr_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Dictionary

# COMMAND ----------

import pandas as pd
data_dict = pd.read_csv('Data_dict.csv')
display(data_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC As a last step for Feature Engineering, we one-hot encoded variables that were deemed to be statistically significantly correlated to a 15 min or greater delay per either Chi-Squared Test (categorical variables) or two sample t-test (numerical variables). This is to account for specific variables that may result resulted in additional features for variables such as day of the week, Origin and Destination airport, and distance. The final table ends up having 1359057 rows and 91 columns for the three month data set. The full 1 year dataset ends up with 5725795 rows and 91 columns.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Tests

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chi-Squared (For categorical variables):
# MAGIC #### Delay by Day of the Week
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20Rate%20by%20Day%20Of%20The%20Week.png?raw=true'>
# MAGIC
# MAGIC
# MAGIC This Chi-Square test will measure the independence of the days of the week based on delays. \
# MAGIC In our case we will be looking at the DAY_OF_WEEK variable, which has 6 degrees of freedom, and we are using alpha = .05. \
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the DAY_OF_WEEK variable. \
# MAGIC In layman's terms, there is statistical evidence to believe that the day of the week affects if a plane will be delayed by 15 or more minutes. 
# MAGIC
# MAGIC #### Delay by Origin (Departure) Airport
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20rate%20by%20Origin%20Airport.png?raw=true'>
# MAGIC
# MAGIC
# MAGIC This Chi-Square test will measure the independence of the origin airports based on delays. \
# MAGIC In our case we will be looking at the ORIGIN_AIRPORT_ID variable, which has 312 degrees of freedom, and we are using alpha = .05. \
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the ORIGIN_AIRPORT_ID variable. \
# MAGIC In layman's terms, there is statistical evidence to believe that the origin airport affects if a plane will be delayed by 15 or more minutes. 
# MAGIC
# MAGIC #### Delay by Destination (Arrival) Airport
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20by%20Arrival%20Airport.png?raw=true'>
# MAGIC
# MAGIC
# MAGIC This Chi-Square test will measure the independence of the destination airports based on delays. \
# MAGIC In our case we will be looking at the DEST_AIRPORT_ID variable, which has 314 degrees of freedom, and we are using alpha = .05.
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the DEST_AIRPORT_ID variable.
# MAGIC In layman's terms, there is statistical evidence to believe that the destination airport affects if a plane will be delayed by 15 or more minutes.
# MAGIC
# MAGIC #### 
# MAGIC
# MAGIC ### T-test (For numerical variables):
# MAGIC #### Delay by Distance
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Distribution%20of%20Flight%20Distances%20by%20Delay%20(or%20No%20Delay).png?raw=true'>
# MAGIC
# MAGIC This two-sample t-test of independence aims to disprove the null hypothesis that delays (of 15 or more minutes) are independent of the distance of the flight. \
# MAGIC In this test we are analyzing the DISTANCE variable divided into two groups (delay or not delay) based on the DEP_DEL15 variable. \
# MAGIC Again, we are using alpha = 0.05 as a threshold for rejecting the null hypothesis. In this case, the test yielded a value of 0.0 and therefore we can reject the null hypothesis. \
# MAGIC In layman's terms, this means that there is statistically significant evidience to believe that the distance of the flight affects whether the flight will be delayed. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC For this phase of the project we focused on experimenting with the 1 year dataset and built two types of models: Logistic Regression and Random Forests. The main experiements that were run were first conducting walk-forward validation on the 1 year dataset with both the Logistic Regression and Random Forest models and looking at the training and validation metrics as the data windowed across a full year. The next set of experiments involved performing hyperparameter tuning on both types of models and checking the chosen metrics after tuning. In the hyperparameter tuning phase we also looked at handling the data imbalance in the dataset with a basic oversampling technique. 
# MAGIC
# MAGIC Once the EDA and feature engineering process was complete, a cleaned version of the 1 year data set was saved to the team Azure blob storage. The general modeling pipeline was then as follows:
# MAGIC
# MAGIC * Split features into categorical and numeric groupings
# MAGIC * For categorical features, feed the features through a StringIndexer followed by a OneHotEncoder 
# MAGIC * For numeric features, feed the features through an imputer. The imputer handled missing values by filling them with the mean value.   
# MAGIC   * Future phases can explore different methods of imputation depending on the feature
# MAGIC * Once all these features are handled, assemble them with a VectorAssembler
# MAGIC * For the logistic regression models, the assembled features were fed through a MinMaxScaler pipeline
# MAGIC * Random forest experiements ingested the vector assembled features directly
# MAGIC
# MAGIC Once the input dataset was fed through the above pipeline, there were a total of 727 features.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loss Functions and Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC There are four main metrics we are analyzing while experimenting with different models and parameters: accuracy, f1 score, weighted precision, and weighted recall. With this project, in the current phase we are looking to optimize all of these metrics. Ultimately, however, F1 Score or Recall are the metrics to focus on. With a highly imbalanced dataset, accuracy can be misleading and with our use case in mind we would like to avoid false negatives as much as possible as informing airlines or customers incorrectly that a flight will not be delayed will cause more issues than if an airlines prepares for a delay when not actually required. Hence recall is prioritized and F1 score is prioritized to take both precision and recall into account. 
# MAGIC
# MAGIC Below are the formulas for each metric:
# MAGIC
# MAGIC \\(Accuracy  = \frac{TP+TN}{TP+TN+FP+FN} \\)
# MAGIC
# MAGIC \\(F1 Score  = \frac{TP}{TP+\frac{1}{2}(FP+FN)} \\)
# MAGIC
# MAGIC \\(Precision  = \frac{TP}{TP+FP} \\)
# MAGIC
# MAGIC \\(Recall  = \frac{TP}{TP+FN} \\)
# MAGIC
# MAGIC Note that in our results section, a weighted average of both classes were used for the precision and recall metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC With the two types of models that were experimented with, there are two main loss functions that were relevent. For Logistic Regression, we use log loss with regularization to guide the tuning process.
# MAGIC
# MAGIC #### Log Loss (With Regularization):
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Log%20Loss.png?raw=true'>
# MAGIC
# MAGIC For Random Forest, we use gini impurity to guide the tuning process:
# MAGIC
# MAGIC #### Gini Impurity
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Gini%20update.png?raw=true'>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiments and Results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Walk-Forward Cross-Validation
# MAGIC The motivation for using walk-forward validation is to ensure there is no data leakage. In this case, using data from later months to train a model and testing on previous months data would cause there be to a bias of having information that would not have been present at the time that the data was collected; hence data *leaks* from later months into earlier months. To mitigate this effect, we look at splitting our training and testing data based on the date. A simple and robust way to split the data is to separate by month, since this analysis is only considering one year of data; our model trains 11 models that test on a month of data, trained on all the previous months in the year (i.e. training on Jan, Feb, and Mar and testing on Apr). The test months are Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov and Dec.
# MAGIC
# MAGIC ### Results
# MAGIC
# MAGIC #### Logistic Regression
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Logistic%20Regression%20Walk-Forward%20Cross-Validation.png?raw=true'>
# MAGIC
# MAGIC From this visual, we can note a few things. Firstly, the accuracy and recall metric are the exact same for every model generated. This can happen when the dataset has a very few number of actual negative outcomes, which is exactly the case in this dataset. Secondly, there is a drop in performance for months 6, 7, 11, and 12 (Jun, Jul, Nov, and Dec respectively). This is interesting because the expectation is that the model performance would increase as more months of data are added to the model, so we hypothesize that because these months are typically when travel frequency is higher (summer breaks, holidays, etc.) that the load put on the airport is increased and the variability of operations is also increased as a result. With that increased uncertainty, the model might perform worse but does give way to add some potentially important variables (holiday indicators, historically more variable times of the year, number of expected flights on that day, etc.)
# MAGIC
# MAGIC #### Random Forest
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Random%20Forest%20Walk-Forward%20Cross-Validation.png?raw=true'>
# MAGIC
# MAGIC From this visual, we note that it follows a very similar pattern to the logistic regression and the majority of the commentary is the same. However, it seemed to perform overall slightly worse than the logistic regression. This could be due to the large amount of one-hot encoded variables that may have led to some splits on data that began to overfit. Typically, the random forest model is fairly robust to overfitting but it is still a consideration we need to make as we develop the models. This is why it is important to test a variety of models.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Fine-Tuning
# MAGIC
# MAGIC Hyperparameter tuning was conducted with logistic regression and random forest models. The training data was the first 3 quarters of the 1 year data set and the last quarter of the 1 year dataset was used as a blind test set. There were two main experiments. The first version is with the raw, cleaned dataset where no data augmentation was done and the second experiment was conducted with an oversampled dataset to handle the outcome variable imbalance. 10 trials were run using hyperopt. 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC In the below table we see the testing data results along with the final hyperparameters after tuning. The overall tuning process for both experiments took ~40 minutes.
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/LogisticRegressionHyperopt.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC Without the oversampled dataset, the fine-tuned Logistic Regression model exhibited extreme bias towards the majority class, predicting 'No Delays' the vast majority of the time. This is likely a result of the dataset's massive class imbalance. With the data augmentation to address this imbalance, all of the metrics decrease significantly-- with the exception of the F1 score. This increase in the F1 score indicates that this LG model trained with oversampling balances between Precision and Recall beter than the model with no oversampling. Important to note is that for both fine-tuned models, the Elastic net and Regularization parameters pushed the majority of the features weights to zero. This is perhaps due to the complex non-linear interactions between features, which LG models cannot capture. This hypothesis is supported by the increase performance by our Random Forest models. In subsequent experiments, further feature engineer and exploring of interaction terms may increase performance. 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Random Forest
# MAGIC
# MAGIC In the below table we see the testing data results along with the final hyperparameters after tuning. The overall tuning process for both experiments took ~40 minutes.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/RF%20Tuned%20Results.png?raw=true'>
# MAGIC
# MAGIC We can see initially, even after hyperparameter tuning we get metrics that essentially align with predicting all "No Delays". This indicates a heavy imbalance in our target variable. So similar to the logistic regression experiments, a follow-up tuning exercise was conducted with an oversampled dataset where the minority class (Delay) was more aligned with the majority class (No Delay). After the data augmentation, we can see our results improve across the board. With the oversampled dataset we see training and validation accuracies of ~80% and ~81% respectively. Furthermore, we can confirm that there are predictions that are being made for delays. One note to consider, only 10 trials were run with hyperopt for the hyperparameter tuning phase. 10 trials took about ~40 min. In follow-up experiments more trials will be explored to see if the model can be further tuned with more searches; however, this must be balanced with the benefit tradeoff of the computing power and overall time sink.
# MAGIC
# MAGIC Ultimately, after tuning and data augmentation the random forest model performs better than the logistic regression. With such a complex set of features and interactions, this aligns with initial expectations. In future phases, we will explore MLP neural networks, gradient boosted decision trees, and potentially other more sophisticated machine learning methods to see if we can improve upon our results. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC Flight delays are a very costly event for airports, airlines, and customers alike. With this project, we aim to explore whether we can create a machine learning model with both given and custom input features that can accurately forecast flight delays to a sufficient degree so that airlines, airport, and customers can adjust their habist and process to avoid these delays. In this phase, we completed full EDA on the provided 3 month and 1 year datasets from 2015. Through the EDA process we identified various features that acted as inputs to our model experiements. We also created some custom features to add to the input model based on some insights gathered from EDA. After this process, we conducted a walk-forward validation on the 1 year dataset with Logistic Regression and Random Forest models. With the walk-forward validation results, we gathered that holiday season may be an important factor in flight delays and plan to add that in future model experiements. Finally, we created some baseline hyperparameter tuned models with Logistic Regression and Random Forest models and evaluated those results. We learned from these initial results that the severe imabalance of the input dataset is hindering the output results. The data imbalance of our target variable (whether a flight was delayed or not) is a major obstacle we will have to consider. In this phase we explored basic oversampling which improved the results somewhat, but future models can explore more sophisticated techniques for data augmentation. Furthermore, we will explore more advanced machine learning methods as well as additional custom features in effort to create a more reliable model that can help predict flight delays. 