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
# MAGIC | Week 2      | Jason        |
# MAGIC | Week 3      | Kushal       |
# MAGIC | Week 4      | Conner        |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Credit Assignment Plan:
# MAGIC #### Phase 1
# MAGIC
# MAGIC | Project Task| Owner |Task Complete? |
# MAGIC | ----------- | ----------- |----------- |
# MAGIC | Setup team blob storage. Write model choice, pipeline, and sucess metric sections | Rodrigo      |Yes|
# MAGIC | Create project timeline and write abstract. Collect and submit databricks notebook      | Kushal       |Yes|
# MAGIC | Create phase leader plan and credit assignment plan. Write section describing dataset   | Jason      |Yes|
# MAGIC | Conduct initial EDA    | Conner        |Yes|

# COMMAND ----------

# MAGIC %md
# MAGIC #### Phase 2
# MAGIC |Project Task|Time Estimate|Jason|Kushal|Rodrigo |Conner |
# MAGIC |:----|:----|:----|:----|:----|:----|
# MAGIC |EDA on all raw tables|2-4 Hrs|X| | | |
# MAGIC |EDA (OPTW)|1-2 Hrs| |X| | |
# MAGIC |Address Missing Data|1 Hr| | |X| |
# MAGIC |Feature Engineering (FE): Address Non-numerical features|3 Hrs for feature engineering total| | | |X|
# MAGIC |FE: Raw data we plan to use| | | | |X|
# MAGIC |FE: Raw data we wish to transform| | | |X| |
# MAGIC |FE: Create final variables that are highly predictive| | |X| | |
# MAGIC |Create ML Pipelines|4 Hrs|X| | | |
# MAGIC |Show and discuss results and findings|2 Hrs|X| | | |
# MAGIC |Fine tune pipeline (grid search)|1 Hr| |X| | |
# MAGIC |Extra Credit (if time allows)|1 Hr|X|X|X|X|
# MAGIC |Clean up Code|0.25 Hr|X | X|X| X|
# MAGIC |Create Presentation and Practice|0.25 Hr|X| X| X|X|
# MAGIC |Clean up notebook and submit project|0.25 Hr| | | |X|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Phase 3
# MAGIC
# MAGIC |Project Task|Time Estimate|Jason|Kushal|Rodrigo |Conner |
# MAGIC |:----|:----|:----|:----|:----|:----|
# MAGIC |Train more sophisticated models - Random Forest|3 Hr|X| | | |
# MAGIC |Train more sophisticated models - Gradient Boosted Decision Tree|3 Hr| |X| | |
# MAGIC |Train more sophisticated models - Multilayer Perceptron Neural Network|3 Hr| | |X| |
# MAGIC |Train more sophisticated models - Extra Credit|3-6 Hr|X|X|X|X|
# MAGIC |Fine tune pipeline with grid search + describe performance|2 Hr| | | |X|
# MAGIC |Test model on 2019 data and discuss performance|1 Hr| | |X| |
# MAGIC |Feature Engineering on new features|2 Hr| |X| | |
# MAGIC |New Directions - Extra Credit|1 Hr|X| | | |
# MAGIC |Clean up Code|0.25 Hr|X| | | |
# MAGIC |Final Presentation|2-3 Hr|X|X | X| X|
# MAGIC |Final Report|4-6 Hr|X| X| X| X|
# MAGIC |Clean up Notebook + Submit|0.25 Hr| |X| | |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project Timeline
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Gantt%20Chart.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC
# MAGIC Flight delays are one of the primary sources of economic losses and overall inconvenience for airlines and passengers during the air travel experience. They can cause major logistical issues which can have a great impact on airline revenue and lower customer satisfaction. As a result, this project is looking to examine existing data on domestic flights, time-series weather data, and weather station data to identify the primary causes of flight delays and ultimately predict flight delays. We will first explore the individual datasets and perform feature engineering to identify raw and transformed variables we would like to use for the machine learning pipelines. We will then work to create various machine learning pipelines such as logistic and linear regression, ensemble models, and MLP neural networks and evaluate the output of these pipelines against historical data to develop reliable predictions. In the end, the goal of this project is to be able to predict flight delays by 2 hours before the planned departure time in order to help airlines and passengers regroup and efficiently address the delay. This effort also aims to help airlines and airports improve delay mitigation and overall flight operation. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data
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
# MAGIC ## Machine Learning Overview
# MAGIC
# MAGIC For our initial model, we will be analyzing an array of models with increasing algorithmic complexity to understand the relationships in the data from multiple perspectives; this will assist in choosing an adequate model for analysis on a larger, more complex engineered feature set. The models we seek to analyze include Logistic Regression, Support Vector Machine, Naive Bayes, and Decision Tree; these algorithms will each provide a unique methodology for creating quantified relationships between variables due to their differing natures of computation. Here, we will describe and justify the use for each of the algorithms, and the associated loss functions.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression
# MAGIC
# MAGIC Logistic regression excels at understanding probabilistic relationships, and we will utilize Binary Cross-Entropy Loss as it is particularly well-suited for the probabilistic nature of the model. This loss measures the disparity between predicted probabilities and actual class labels and quantifies the difference between the predicted probability of a positive instance (delayed flight) and the actual binary label (1 for delayed, 0 for not delayed). This loss function penalizes misclassifications with increasing severity as the predicted probability diverges from the true class label, making it particularly sensitive to instances where the model is highly confident in an incorrect prediction. This property aligns well with the objectives of logistic regression, which seeks to maximize the likelihood of observing the true class labels given the input features.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Other Models
# MAGIC We plan to execute a similar cross-validation sequencing with several other models, taking advantage of the different algorithmic nuances that come with each model. The other models we would like to experiment with include Support Vector Machines, Naive Bayes, and Decision Trees. Each of these models will contain a different loss metric that will allow for the proper minimization of loss for the given algorithm; SVM will utilize hing loss, Naive bayes similar to logistic regression will use Binary Cross-Entropy, and decision trees will utilize the greedy Gini Impurity loss function. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Overview
# MAGIC
# MAGIC The pipeline will be a generalized ETL process that will extract the raw data from large-file storage in the Azure blob and transform it into a dataset that will contain the features we will look at in the first round of modeling. Transformations will include datatype cleaning, null-value analysis, joins, categorical variable encoding, numerical variable scaling, and imputations for variables that are missing less than 10% of the data. This data will then be piped in parallel to each of the models to begin k-fold cross-validation based training processes. These models and pipes will be saved into our file storage for later use. Then, for each model, the classification model evaluation metrics will be calculated: accuracy, precision, recall, and F-score. During this process, we will also optimize the hyperparameters through a k-fold cross-validation optimization process (Bayesian optimization, Random Search, or Grid Search). These optimized models will be saved to be used for test evaluation.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/ETL%20Pipeline.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Success Metrics
# MAGIC
# MAGIC In our model, we will seek to correctly classify all positive (delay) outcomes and minimize the number of false positives. If we classify a flight as a delay which causes someone to show up 15 minutes late and miss their flight, that would be much worse than saying a flight is on time and someone having to wait a little longer. On the other hand, if we predict a flight to not be delayed that is actually on time (false negative), it may cause issues on the tarmac at the airport like congestion that ultimately lead to more delays. Our experimentation phase will include a cost-benefit analysis of what might happen in the case of favoring minimizing false positives or false negatives. From a quantitative standpoint, this would involve maximizing either the precision or recall respectively. To calculate precision, we divide the number of true positives (predicted delay and actual delay) by the total number of positive predictions (predicted delays). To calculate recall, we divide the number of true positives (predicted delay and actual delay) by the sum of true positives (predicted delay and actual delay) and false negatives (predicted not delay and actual delay).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preliminary Visual EDA
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Delay%20Types%20Bar%20Chart.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC Here we have a grouped bar chart that displays the average delay in minutes by month for each of the delay categories: Weather, Carrier, NAS, Late-Aircraft. For each of the first 3 months, the delays are relatively consistent with the exception of weather delays, which appears to fluxuate sigificantly between each month. It is clear from this plot that weather features will play a crucial role in our feature engineering process. More EDA will be necessary in order to determine the relevance of these other delay types. For instance, will certain airports have significant carrier delays relative to the average carrier delay?

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Windspeed-Rainfall-Visibility.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC The plots above display the average daily windspeed, Rainfall, and visibility for the first 3 months of 2015. Even within a 3 month range, each of the time-series, particulary windspeed and visibilty, fluxuate significantly and display clear patterns between the different months. For instance, in the month of March visibility increases and varys significantly more compared to the month of February. It is clear that the weather data we are working with is highly granular, and indicates that accounting for seasonality will be crucial. 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Weather%20Code%20-%20On%20Time.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Weather%20Code%20-%20Delay.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC These two table display the top 10 most common weather code entries by relative count for delayed and non-delayed flights. This is a 'Quick and Dirty' EDA approach to understanding the significance of the weather type codes in the data. In our extensive EDA, we will need to clean this data and seperate out each weather code individually. Even without data cleaning, this data highlights clear relationships between weather codes and whether or not a flight was delayed. Most sigificantly, the code '-SN:03 |SN|' which indicates snow, has a relative frequency almost double for delayed flights compared to non-delayed flights. It is clear that this cleaned data will play a crucial role in our weather features. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC Based on initial analysis, there are many different avenues we can explore to build our machine learning pipelines and generate accurate predictions. Next steps will include conducting a full EDA exercise on all of the data and also potentially exploring additional joined datasets to identify patterns and features. Once EDA and feature engineering is complete, we can begin to build our initial machine learning pipelines for baseline linear/logistic regression and some ensemble models. After the simpler models, we can expand into neural networks and other more advanced techniques. Throughout this process we will tune and experiment with different models, features, etc. Ultimately, we will decide upon our "best" model to present to our client. 

# COMMAND ----------

