# Databricks notebook source
# MAGIC %md
# MAGIC # Up in the Air: Predicting Flight Delays

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Team 2-1
# MAGIC
# MAGIC ### Jason Frankel (jasonfrankel@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Jason%20Headshot.png?raw=true'>
# MAGIC
# MAGIC ### Rodrigo Cochran (rodrigocochran@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Rodrigo%20Headshot.png?raw=true'>
# MAGIC
# MAGIC ### Conner Davis (connerdavis@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Conner%20Headshot.png?raw=true'>
# MAGIC
# MAGIC ### Kushal Gourikrishna (kgourikrishna@berkeley.edu)
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%201/Kushal%20Headshot.png?raw=true'>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##Phase Leader Plan
# MAGIC | Project Week | Phase Leader |
# MAGIC | ----------- | ----------- |
# MAGIC | Week 1      | Rodrigo      |
# MAGIC | Week 2      | Jason        |
# MAGIC | Week 3      | Kushal       |
# MAGIC | Week 4      | **Conner**        |

# COMMAND ----------

# MAGIC %md
# MAGIC ##Updated Credit Assignment Plan
# MAGIC #### Phase 3
# MAGIC
# MAGIC | Project Task | Owner | Task Complete? | Start Date | End Date | Effort |
# MAGIC | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
# MAGIC | Setup and run Walk-Forward Cross Validation. Add page rank feature to model inputs. Presentation and final report contribution | Rodrigo      |Yes| April 8th | April 16 | 40 hours |
# MAGIC | Setup MLP pipelines and run experiments (hyperopt). Presentation and final report contribution   | Kushal       |Yes| April 8th | April 16 | 40 hours |
# MAGIC | Additional feature engineering (Holidays feature, airline ranking feature, etc) and EDA. Adding 2020-2022 dataset (extra credit) Presentation and final report contribution| Jason      |Yes| April 8th | April 16 | 40 hours |
# MAGIC | Setup GBT and RF pipelines and run experiments (hyperopt). Presentation and final report contribution   | Conner        |Yes| April 8th | April 16 | 40 hours |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC
# MAGIC This project aims to conduct a comprehensive analysis leveraging existing datasets encompassing domestic flight records, time-series weather data, and meteorological station data to elucidate the principal factors contributing to flight delays and subsequently forecast delays exceeding 15 minutes. Initially, we undertook an in-depth exploration of integrated datasets, meticulously engineering features with discernible correlations to flight delays, thus furnishing a robust foundation for predictive modeling. Employing Multilayer Perceptron, Random Forest, and Gradient Boosted Decision Tree algorithms within this analytical framework, we constructed prediction pipelines to evaluate their efficacy in delivering accurate forecasts.
# MAGIC
# MAGIC To mitigate the risk of data leakage and ascertain the temporal resilience of our models, we executed walk-forward validation procedures. Furthermore, a series of methodical experiments ensued, encompassing rigorous hyperparameter optimization, application of strategies to address data imbalance, and dimensionality reduction techniques. Our findings reveal that the Gradient Boosted Tree Model, without oversampled data, emerges as the optimal pipeline, exhibiting approximately 85.6% accuracy and 83.2% F1-score. This represents a marginal enhancement over a baseline model that uniformly predicts "No Delay" for all instances.
# MAGIC
# MAGIC Future research endeavors may delve into advanced methodologies for addressing data imbalance, augmenting dataset volume, and exploring additional neural network architectures and sophisticated machine learning models to further refine predictive accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Description
# MAGIC
# MAGIC The datasets we are working with here include 3 raw datasets, in additino to a pre-joined dataset. 
# MAGIC
# MAGIC ### Domestic Flight Data
# MAGIC The first raw dataset is domestic flight data from the bureau of transportation statistics (BTS). The data is already pre-divided into three sections: 1st quarter of 2015, first two quarters of 2015, and 2015-2019 data. As to be discussed in the Exploratory Data Analysis (EDA) section, it contains fields such as airline, flight number, departure, summary of the flight, and arrival information, including whether there was a delay and what was the cause of said delay.
# MAGIC
# MAGIC ### Global Weather Data
# MAGIC The second raw dataset is global weather data from the National Oceanic and Atmospheric Administration (NOAA) repository with hourly, daily, and monthly summaries of weather information from 2015-2021. This contains the locations of the weather stations used in the data as well as the time, with weather noted including temperature, wind, precipitation type + amount, air pressure, and sun rise + set to name a few. Refer to the EDA section below for further insights and details.
# MAGIC
# MAGIC ### Weather Stations Data
# MAGIC The third and last raw dataset provides comprehensive information regarding all weather stations utilized in the aforementioned weather dataset. This dataset proves particularly valuable due to its inclusion of detailed attributes for each weather station, including their spatial relationships with the primary station under consideration and the distance to neighboring stations. This will be useful in assessing the impact of weather conditions proximate to the arrival destination or elsewhere in the country on flight delays at a specific location.
# MAGIC
# MAGIC ### Final Dataset
# MAGIC Finnally, we combined all aforementioned datasets to form the **On Time Performance and Weather (OPTW)** joined dataset. The OPTW dataset is available in three versions corresponding to different time periods (OPTW for Q1 2015, H1 2015, and 2015-2019) and combines flight, weather, and station data to produce a unique record for each flight (date + tail num + airline). This composite dataset encompasses comprehensive flight information derived from flight data, augmented with relevant weather data sourced from the closest weather station to the departure location, captured four hours preceding the scheduled departure time. An exploratory question would investigate if supplemental data had additional predictive power (weather at arrival station, other stations en route, etc.).

# COMMAND ----------

# MAGIC %md
# MAGIC ## ETL Workflow Overview
# MAGIC
# MAGIC The overall workflow was a generalized ETL process that extracted the raw data from large-file storage in the Azure blob and transformed it into a dataset that will contain the features we analyzed in the first round of modeling. Transformations will included datatype cleaning, null-value analysis, categorical variable encoding, numerical variable scaling, and imputations for variables that are missing less than 10% of the data. The data was then piped into various model experiements explained later in this report. Then, for each model, the classification model evaluation metrics were calculated: accuracy, precision, recall, and F-score. During this process, we also optimized hyperparameters through Bayesian Optimization.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/ETL%20Workflow.png?raw=true'>
# MAGIC
# MAGIC **Figure 1**. *ETL Pipeline Diagram*: This diagram visualizes the steps necessary for transforming the data for analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Histograms of Subset of Variables Side-by-Side
# MAGIC
# MAGIC To gain an understanding of the distrubtion of our variables, we created visualizations of the variables we were most curious about.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Histograms_New.png?raw=true'>
# MAGIC
# MAGIC **Figure 2**. *Distributions For Key Variables*: This set of histograms visualizes the distribution for several key variables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delay Rate Analysis
# MAGIC
# MAGIC Below is an example plot showing delay rate against the unique carrier feature. This initial EDA was done with all features to understand the realtionship with delay rate for potential input features. In the feature engineering section, there are further examples of this analysis.
# MAGIC  
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Delay%20Rate%20-%20Unique%20Carrier.png?raw=true'>
# MAGIC
# MAGIC **Figure 3**. *Delay Rate by Carrier*: This graph shows the delay rate for each unique carrier.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC In the flight data preprocessing phase, we began by establishing a unique identifier (UID) for each flight, which was formed by combining the flight number, carrier ID, and date. We then created two time-based variable using window functions: first to determine whether a plane in any of the preceding flights to the current flight was in the air 2 hours before the current expected takeoff time and then if said plane that was in the air was delayed.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Time-based Variables
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Time%20Variable%20Histogram.png?raw=true'>
# MAGIC
# MAGIC **Figure 4a.** *Airborne Distribution of Previous Flight, 2 Hour Time Window*: This graph shows the distribution of flights of whether the previous flight was airborne, where previous means 2 hours before the expected takeoff of the current flight.
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20Rate%20by%20Time%20Variable.png?raw=true'>
# MAGIC
# MAGIC **Figure 4b.** *Delay Rate by Airborne Previous Flight (Tail Num)*: This chart shows the delay rate by whether or not the previous flight (the tailing flight) was airborne 2 hours before the expected takeoff of the current flight.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Flight%20Delayed%20Histogram.png?raw=true'>
# MAGIC
# MAGIC **Figure 4c.** *Previous Flight Delay Distribution*: This chart shows the distribution of delay (or not delay) of the previous flight.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Delay%20Rate%20by%20Delay%20Variable.png?raw=true'>
# MAGIC
# MAGIC **Figure 4d.** *Delay Rate byPrevious Flight Delay*: This chart shows the delay rate of the current flight, by whether the previous flight was delayed or not delayed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph Algorithm - Pagerank
# MAGIC We decided to use the pagerank of the origin airport where vertices are the destination airport to gain an understanding of the connectedndess of that airport. We hypothesize that the airports with higher connectedness may have a higher probability of delay.
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Pagerank%20by%20Airport.png?raw=true'>
# MAGIC
# MAGIC **Figure 5a**. *Pageranks by Airport*: This chart shows that the higher ranked airports are significantly higher ranked than the lower ranked airports; meaning that the airports that are well connected are significantly more connected than lower ranked airports.  \
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Top%2010%20Airports%20-%20Pagerank.png?raw=true'>
# MAGIC
# MAGIC **Figure 5b**. *Top 10 Airports by Pagerank*: Airports like ATL (Atlanta), ORD (Chicago O'Hare), and DFW (Dallas Fort Worth) are the top 3 ranked airports and this makes sense as they are considered hubs for several airlines. They are also geographically in the center of many large areas, and near metropolitan cities.
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Bottom%2010%20Airports%20-%20Pagerank.png?raw=true'>
# MAGIC
# MAGIC **Figure 5c**. *Bottom 10 Airports by Pagerank*: The airports in the bottom 10 represent low population areas that are often on the edge of the contigious United States. These 10 airports all have a very low pagerank that is nearly the same, likely that the pagerank converged for these "semi-dangling" nodes (airports).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Additional Variables
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/WSJ_Rankings.png?raw=true'>
# MAGIC
# MAGIC **Figure 6a**. *Delay Rate by Wall Street Jouranl Ranking*: This chart shows delay rates of the top 10 airports ranked by the Wall Street Journal in 2015.
# MAGIC
# MAGIC <img src = 'https://github.com/kgourikrishna/w261_finalproject/blob/main/Holidays.png?raw=true'>
# MAGIC
# MAGIC **Figure 6b**. *Delay Rate by Holiday*: This chart shows delay rates of popular holidays in the United States.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature Analysis
# MAGIC
# MAGIC ### Refinement and Cleaning
# MAGIC We conducted a rigorous preprocessing phase to refine our dataset for correlational analysis. This involved the systematic elimination of redundant variables, such as arrival-based information, which cannot be reliably ascertained prior to flight departure. We also excluded data pertaining to cancelled flights, non-numeric fields, and daily/monthly weather features, deeming them unnecessary in light of the availability of more granular hourly weather data. 
# MAGIC
# MAGIC ### Weather Codes and Sky Codes
# MAGIC Furthermore, we meticulously crafted a set of boolean variables, comprising 43 key weather codes and 8 sky codes, to enhance the predictive capacity of our model.
# MAGIC
# MAGIC ### External Data
# MAGIC In addition to these measures, we augmented our OPTW dataset by integrating external data sources. Specifically, we included the top 10 airlines as ranked annually by the Wall Street Journal (2015 rankings: https://www.wsj.com/articles/the-best-and-worst-airlines-of-2015-1452709957). This alignment with industry rankings offers valuable context for our analysis. Moreover, we enriched our dataset by linking dates to a repository of federal holidays, facilitating insights into potential holiday-related influences on flight delays. For detailed delay rates corresponding to airline rankings and holidays, refer to the provided breakdown below.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Matrix
# MAGIC Prior to applying one-hot encoding for variable representation, we executed a comprehensive set of analytical procedures to prepare the dataset. These included a correlation analysis, variable description, frequency analysis, pairwise input-output examination, and the construction of a comprehensive data dictionary, all documented for transparency and reproducibility, as presented below.
# MAGIC
# MAGIC Among the most important insights obtained from this preparation phase, our analysis underscored notable disparities across carriers, airports, and precipitation levels concerning their propensity to influence flight delay occurrences significantly. This discernment assisted our analysis in explaining the reasons for delay occurrences.
# MAGIC
# MAGIC Below we have included a correlation matrix that provides a numerical representation of the linearity of the relationships among our input variables.

# COMMAND ----------

#### Correlation Matrix
# <!-- 
# <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%202/Correlation%20Matrix.png?raw=true'> -->

import pandas as pd
corr_matrix = pd.read_csv('correlation_matrix_eda.csv')
display(corr_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Dictionary
# MAGIC To bring transparency to the variable definitions, we provide here a data dictionary that maps varibale names to a description and its data type.

# COMMAND ----------

import pandas as pd
data_dict = pd.read_csv('Data_Dict.csv')
display(data_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2020-2022 Dataset
# MAGIC We also had a cleaned up version of the same flight data we used in our model for more recent time periods (2020-2022). A small snippet of the data is shown below. A full version is stored on our team storage blob. 

# COMMAND ----------

import pandas as pd
extra_credit = pd.read_csv('extra_credit.csv')
display(extra_credit)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Statistical Tests
# MAGIC
# MAGIC #### Chi-Squared (For categorical variables):
# MAGIC ##### Delay by Day of the Week
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Delay%20by%20day%20of%20the%20week.png?raw=true'>
# MAGIC
# MAGIC **Figure 7a.** *Delay by Day of the Week*: This graph shows the delay rate for each day of the week.
# MAGIC
# MAGIC This Chi-Square test will measure the independence of the days of the week based on delays. \
# MAGIC In our case we will be looking at the DAY_OF_WEEK variable, which has 6 degrees of freedom, and we are using alpha = .05. \
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the DAY_OF_WEEK variable. \
# MAGIC In layman's terms, there is statistical evidence to believe that the day of the week affects if a plane will be delayed by 15 or more minutes. 
# MAGIC
# MAGIC ##### Delay by Origin (Departure) Airport
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Top%2010%20Highest%20Delays%20by%20Origin.png?raw=true'>
# MAGIC
# MAGIC **Figure 7bi.** *Delay by Origin Airport*: This graph shows the top 10 delay rates by origin airport.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Top%2010%20Lowest%20Delays%20by%20Origin.png?raw=true'>
# MAGIC
# MAGIC **Figure 7bii.** *Delay by Origin Airport*: This graph shows the bottom 10 delay rates by origin airport.
# MAGIC
# MAGIC
# MAGIC This Chi-Square test will measure the independence of the origin airports based on delays. \
# MAGIC In our case we will be looking at the ORIGIN_AIRPORT_ID variable, which has 312 degrees of freedom, and we are using alpha = .05. \
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the ORIGIN_AIRPORT_ID variable. \
# MAGIC In layman's terms, there is statistical evidence to believe that the origin airport affects if a plane will be delayed by 15 or more minutes. 
# MAGIC
# MAGIC ##### Delay by Destination (Arrival) Airport
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Top%2010%20highest%20delays%20by%20destination%20airport.png?raw=true'>
# MAGIC
# MAGIC **Figure 7ci.** *Delay by Destination Airport*: This graph shows the top 10 delay rates by destination airport.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Top%2010%20lowest%20delays%20by%20destination%20airport.png?raw=true'>
# MAGIC
# MAGIC **Figure 7cii.** *Delay by Destination Airport*: This graph shows the bottom 10 delay rates by destination airport.
# MAGIC
# MAGIC This Chi-Square test will measure the independence of the destination airports based on delays. \
# MAGIC In our case we will be looking at the DEST_AIRPORT_ID variable, which has 362 degrees of freedom, and we are using alpha = .05.
# MAGIC To be significant, the P-value must be smaller than alpha (.05) so our test result of 0.0 is statistically signifcant, meaning we can reject the null hypothesis that the DEP_DEL15 variable is independent of the DEST_AIRPORT_ID variable.
# MAGIC In layman's terms, there is statistical evidence to believe that the destination airport affects if a plane will be delayed by 15 or more minutes.
# MAGIC
# MAGIC #### T-test (For numerical variables):
# MAGIC ##### Delay by Distance
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Delay%20by%20Distance.png?raw=true'>
# MAGIC
# MAGIC **Figure 7d.** *Delay by Distance*: This graph shows the ratio of delay by distance of flight.
# MAGIC
# MAGIC This two-sample t-test of independence aims to disprove the null hypothesis that delays (of 15 or more minutes) are independent of the distance of the flight. \
# MAGIC In this test we are analyzing the DISTANCE variable divided into two groups (delay or not delay) based on the DEP_DEL15 variable. \
# MAGIC Again, we are using alpha = 0.05 as a threshold for rejecting the null hypothesis. In this case, the test yielded a value of 0.0 and therefore we can reject the null hypothesis. \
# MAGIC In layman's terms, this means that there is statistically significant evidience to believe that the distance of the flight affects whether the flight will be delayed. 
# MAGIC
# MAGIC ##### Delay by Pagerank
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Pagerank%20T-test.png?raw=true'>
# MAGIC
# MAGIC **Figure 7e.** *Delay by Pagerank*: This graph shows the ratio of delay by the pagerank of the origin airport.
# MAGIC
# MAGIC This two-sample t-test of independence aims to disprove the null hypothesis that delays (of 15 or more minutes) are independent of the pagerank of the origin airport. \
# MAGIC In this test we are analyzing the pagerank variable divided into two groups (delay or not delay) based on the DEP_DEL15 variable. \
# MAGIC Again, we are using alpha = 0.05 as a threshold for rejecting the null hypothesis. In this case, the test yielded a value of 0.0 and therefore we can reject the null hypothesis. \
# MAGIC In layman's terms, this means that there is statistically significant evidience to believe that the pagerank of the origin airport of the flight affects whether the flight will be delayed. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Final Dataset Considerations
# MAGIC
# MAGIC As a last step for data preparation, we one-hot encoded variables that were deemed to be statistically significantly dependent to a 15 min or greater delay per either Chi-Squared Test (categorical variables), and additionally directly included numerical variables that resulted in statistically significnat dependence from the two sample t-test. 
# MAGIC
# MAGIC The final dataset for modeling in phase 3 consists of 31,212,426 rows. After eliminating redundant columns, including ORIGIN and DEST IDs, DISTANCE GROUP, etc., the dataset is composed of 88 columns, before one-hot encoding.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling Pipelines

# COMMAND ----------

# MAGIC %md
# MAGIC For this phase of the project we focused on experimenting with the  5 year dataset (2015-2019) and built four types of models: Two variations of Multilayer Perceptrons, Random Forest, and Gradient Boosted Decision Trees The main experiments that were run were first conducting walk-forward validation on the 5 year dataset with all models and looking at the training and validation metrics as the data windowed across multiple years. The next set of experiments involved performing hyperparameter tuning on both types of models and checking the chosen metrics after tuning. We also once again looked at handling the data imbalance in the dataset with a basic oversampling technique. 
# MAGIC
# MAGIC Once the EDA and feature engineering process was complete, a cleaned version of the 5 year dataset was saved to the team Azure blob storage. The general modeling pipeline was then as follows:
# MAGIC
# MAGIC * Split features into categorical and numeric groupings
# MAGIC * For categorical features, feed the features through a StringIndexer followed by a OneHotEncoder 
# MAGIC * For numeric features, feed the features through an imputer. The imputer handled missing values by filling them with the mean value.   
# MAGIC * Once all these features are handled, assemble them with a VectorAssembler
# MAGIC * For the MLP models, the assembled features were fed through a MinMaxScaler pipeline
# MAGIC * Random forest and GBT experiments ingested the vector assembled features directly
# MAGIC * For some experimentation at the end, PCA with 12 principal components was added to the MLP pipelines to see if reducing the input feature dimension improved results
# MAGIC
# MAGIC Once the input dataset was fed through the above pipeline, there were a total of 884 features.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loss Functions and Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC There are four main metrics we are analyzing while experimenting with different models and parameters: accuracy, f1 score, weighted precision, and weighted recall. With this project, in the current phase we are looking to optimize all of these metrics. Ultimately, however, F1 Score or Recall are the metrics to focus on. With a highly imbalanced dataset, accuracy can be misleading and with our use case in mind we would like maximize the F1 Score. 
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
# MAGIC With the two types of models that were experimented with, there are two main loss functions that were relevent. For the MLP models and Gradient Boosted Trees we use Mean Squared Error.
# MAGIC
# MAGIC #### MSE (Mean Squared Error):
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/MSE.png?raw=true'>
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
# MAGIC
# MAGIC #### Motivation: Data Leakage
# MAGIC
# MAGIC The motivation for using walk-forward validation is to ensure there is no data leakage. In this case, using data from later time periods to train a model and testing on previous time periods’ data would cause there to be a bias of having information that would not have been present at the time that the data was collected to predict on; hence data *leaks* from later time periods into earlier time periods. To mitigate this effect, we look at splitting our training and testing data based on the date. A simple and robust way to split the data is to separate by year, since this analysis is considering multiple years of data; our model trains 4 models that test on one year of data, trained on all the previous years (i.e. training on 2015, 2016, 2017 and testing on 2018). The test months are 2016, 2017, 2018, and 2019.
# MAGIC
# MAGIC ### Results
# MAGIC
# MAGIC #### Random Forest
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Random%20Forest%20Walk-Forward%20Cross-Validation.png?raw=true'>
# MAGIC
# MAGIC **Figure 8a**. *Walk-Forward Cross-Validation Results: Random Forest*: Visual representation of the results of the walk-forward cross validation for the Random Forest model.
# MAGIC
# MAGIC From this visual, we can note a few things. Firstly, the accuracy and recall metric are the exact same for every model generated. This can happen when the dataset has a very few number of actual negative outcomes, which is exactly the case in this dataset. Secondly, there is a drop in performance for every year going forward across every metric. This is interesting because the expectation is that the model performance would increase as more years of data are added to the model, so we hypothesize that there is a trend component which our model is not capturing that could explain the increased variability over time. With that increased uncertainty, the model might perform worse but does give way to add some potentially important variables related to predictive trends of numerical variables.
# MAGIC
# MAGIC #### Gradient Boosted Trees
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/Gradient%20Boosted%20Trees%20Walk-Forward%20Cross-Validation.png?raw=true'>
# MAGIC
# MAGIC **Figure 8b**. *Walk-Forward Cross-Validation Results: Gradient Boosted Trees*: Visual representation of the results of the walk-forward cross validation for the Gradient Boosted Trees model.
# MAGIC
# MAGIC From this visual, we note that it follows a very similar pattern to the Random Forest and the majority of the commentary is the same. The only difference is that the Gradient Boosted Trees showed an increase in performance from the 2016 test to the 2017 test. Overall, it seemed to perform better than the Random Forest. This is likely due to the nature of the algorithm where it corrects erroneous predictions after each iteration, and is as not as prone to overfitting on unbalanced datasets.
# MAGIC
# MAGIC #### MLP V1
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/MLP%20V1%20Walk-Forward%20Cross-Validation.png?raw=true'>
# MAGIC
# MAGIC **Figure 8c**. *Walk-Forward Cross-Validation Results: Multi-Layer Perceptron V1*: Visual representation of the results of the walk-forward cross validation for the first version of the Multi-Layer Perceptron model.
# MAGIC
# MAGIC Our first version of the Mult-iLayer Perceptron followed the structure of a single input layer with the number of features as the number of neurons, 1 hidden layer with 5 neurons, and a binary output layer (as we are predicting a binary output). Each layer has a sigmoid activation function, and the output layer has softmax activation. From this graph we can see that it follows a similar pattern to the gradient boosted trees, except that it has a much sharper drop off in the 2019 test. This big drop in precision means that there are a higher number of false positives, which can happen in a highly unbalanced dataset like the one we used.
# MAGIC
# MAGIC #### MLP V2
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/MLP%20V2%20Walk-Forward%20Cross-Validation.png?raw=true'>
# MAGIC
# MAGIC **Figure 8d**. *Walk-Forward Cross-Validation Results: Multi-Layer Perceptron V2*: Visual representation of the results of the walk-forward cross validation for the second version of the Multi-Layer Perceptron model.
# MAGIC
# MAGIC The second version of the Multi-Layer Perceptron used a similar setup as the first, but with 2 hidden layers with 10 neurons each. Though the algorithm was more advanced, it performed worse overall, likely due to overfitting. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Tuning and Results
# MAGIC
# MAGIC We ran multiple experiments across the different model types to help determine a "best" model for this project. The 2019 flight/weather data was used as a blind test set to evaluate the models that were trained on the 2015-2018 data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest
# MAGIC In each of the below tables we see the testing data results along with the final hyperparameters after tuning. The wall times for each experiment is as follows:
# MAGIC * Oversampled RF: ~4 minutes
# MAGIC * Tuned RF with Hyperopt (No Oversampling): ~40 minutes 
# MAGIC * Tuned RF with Hyperopt (Oversampling): ~ 45 minutes 
# MAGIC <br />  
# MAGIC <br />  
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/RF.png?raw=true'>. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Gradient Boosted Trees
# MAGIC
# MAGIC In each of the below tables we see the testing data results along with the final hyperparameters after tuning. The wall times for each experiment is as follows:
# MAGIC * Oversampled Gradient Boosted Trees: ~5 minutes
# MAGIC * Tuned Gradient Boosted Trees with Hyperopt (no Oversampling): ~40 minutes 
# MAGIC * Tuned Gradient Boosted Trees with Hyperopt (Oversampling): ~45 minutes 
# MAGIC <br />  
# MAGIC <br />  
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/GBT.png?raw=true'>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The results from tree-based models, Random Forest and Gradient Boosted Trees (GBT), follow similar patterns. Each baseline model with oversampling are outperformed by their hyperopt tuned counterpart. Interestingly, we can see that in both cases the non-oversampled hypertuned models perform the best by a significant margin. For RF, non-oversampled outperforms its oversampled counterpart by ~.17, an increase of ~ 20%. Similarly for GBT, the non-oversampled model outperforms by ~.13, a ~15% increase. 
# MAGIC
# MAGIC  This result contradicts our experiments in Phase 2, which showed that oversampling the minority class stopped our models from simplying predicting the majority class across the board. However, these more sophisticated models-- particuarly the GBT-- combined with our additional feature engineer in Phase 3, don't exhibit this same tendency. Our results indicate that oversampling introduces bias and thus reduces the overall performance. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### MLP Version 1
# MAGIC ##### Input Layer (884 Input Features) - 1 Hidden Layer (5 Neurons) - Output Layer (Binary Output)
# MAGIC
# MAGIC ##### Each layer has sigmoid activation function, output layer has softmax.
# MAGIC
# MAGIC In each of the below tables we see the testing data results along with the final hyperparameters after tuning. The wall times for each experiment is as follows:
# MAGIC * Oversampled MLP: ~6 minutes
# MAGIC * Tuned MLP with Hyperopt: ~1 hour
# MAGIC * MLP with PCA: ~5 minutes
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/MLP%20Version%201.png?raw=true'>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### MLP Version 2
# MAGIC ##### Input Layer (884 Input Features) - 2 Hidden Layers (10 Neurons each) - Output Layer (Binary Output)
# MAGIC
# MAGIC ##### Each layer has sigmoid activation function, output layer has softmax.
# MAGIC
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/MLP%20Version%202.png?raw=true'>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### MLP Commentary
# MAGIC
# MAGIC ##### Version 1
# MAGIC Beginning with the first version of the MLP, we see initially that using oversampled data with the MLP shows some reduced performance compared to the hyperparameter tuned version of the model using the raw data. In this case, it seems as though the introduced oversampling bias reduces the quality of the overall results. With the hyperparameter tuned MLP, we get solid results across the board with each metric being around 85%. 
# MAGIC
# MAGIC ##### Version 2
# MAGIC Looking at the second MLP version, we see a similar reduced performance with the oversampled data. However, with the introduction of a second hidden layer and increased neurons we get slightly better performance with oversampled data than the MLP version 1. Nevertheless, the hyperparameter tuned model with raw data still gives us the best results similar to MLP version 1.
# MAGIC
# MAGIC ##### Principal Component Analysis
# MAGIC With the last experiment involving PCA, we once again see a slight reduction in performance in both MLP versions compared to the hyperparameter tuned model. The thought process behind this experiment was that with 884 input features we may have some noise that are impacting the MLP performance. By reducing the dimensionality we could try to preserve the information contained within the input features but reduce overall complexity. However, one factor to consider is that our input feature set has many one hot encoded variables. When using PCA for one hot encoded variables it may not be very effective. PCA tries to capture the majority of the explained variance of the input feature set while removing redundant information. But with one-hot encoded variables they can either be 0 or 1 so trying apply PCA to these varaibles may not be very useful. Future experiments can look at different techniques to reduce one-hot encoded variables along with the numeric continuous variables in our input set. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Pipeline and Gap Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Best Pipeline: Gradient Boosted Trees (without oversampling) 
# MAGIC
# MAGIC ##### Hyperparameters:
# MAGIC
# MAGIC {max_bins: **57.0**, max_depth: **7.0**, minInstancesPerNode: **8.0**} 
# MAGIC
# MAGIC ##### Evaluation Metrics:
# MAGIC * Accuracy: 0.856 
# MAGIC * F1 Score: 0.832 
# MAGIC * Precision (weighted): 0.842 
# MAGIC * Recall (weighted): 0.852 
# MAGIC
# MAGIC ##### Confusion Matrix ####
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/cf.png?raw=true'> 
# MAGIC
# MAGIC **Figure 9**. *Confusion Matrix*: This chart shows the numerical distribution of true positives, true negatives, predicted positives, and predicted negatives yielded by the predictions of the model.
# MAGIC
# MAGIC #### Gap Analysis
# MAGIC
# MAGIC ##### Phase 2 Results
# MAGIC In the below graph from Phase 2, we see that all of the predictions made were below the threshold value of 0.5, meaning that every outcome was predicted to be 0 (not delayed). This could have been due to many factors, but most importantly the severe class imblance that was present in the training data. 
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/RF_model.png?raw=true'> 
# MAGIC
# MAGIC **Figure 10a**. *Phase 2 Distribution of Predicted Class Probabilities*: This graph shows the distribution of predicted probabilities for the output classes for the Phase 2 GBT model.
# MAGIC
# MAGIC ##### Phase 3 results
# MAGIC In the following graph from Phase 3, we observe a significantly expanded range of predicted probabilities both surpassing and falling below the designated threshold. This expansion signifies the model's capability to predict instances of delay as well as non-delay occurrences, showcasing an advancement over the baseline model, which uniformly predicted absence of delay.
# MAGIC
# MAGIC Additionally, several key differences were observed between the distributions of predicted delay and non-delay:
# MAGIC
# MAGIC * The distribution of the predicted non-delay class exhibits a pronounced right skew, indicating a high probability of correct non-delay predictions.
# MAGIC * Conversely, the distribution of the predicted delay class appears relatively uniform, suggesting a uniform probability of correct delay predictions.
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/GBT%20Histogram%20of%20Probabilities%20Phase%203.png?raw=true'> 
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Figure 10b**. *Phase 3 Distribution of Predicted Class Probabilities*: This graph shows the distribution of predicted probabilities for the output classes for the Phase 3 GBT model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Examaining the false negative predictions more closely (figure 10c), the largest group of misclassifications, we can see that the predicted probabilities have a sharp right skew, where most fall somewhere between .1 - .2 area. This significant gap between where most false negative predicted probabilities lie and the .5 threshold, likely indicate our current feature set is insufficient to mitigate false negatives. 
# MAGIC
# MAGIC
# MAGIC In order to improve on these results, additional feature engineering, consulting of additional data sources, and a more robust solution to the class imbalance problem is likely needed.  
# MAGIC
# MAGIC
# MAGIC <img src ='https://github.com/kgourikrishna/w261_finalproject/blob/main/Phase%203/GBT%20Histogram%20of%20Probabilities%20Phase%202.png?raw=true'> 
# MAGIC
# MAGIC
# MAGIC **Figure 10c**. *Phase 3 Distribution of Predicted Class Probabilities For False Negatives*: This graph shows the distribution of predicted probabilities of false negatives for the Phase 3 GBT model.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and Next Steps
# MAGIC ### Review of Objective
# MAGIC Flight delays are a very costly event for airports, airlines, and customers alike. With this project, we aim to explore whether we can create a machine learning model with both given and custom input features that can accurately forecast flight delays to a sufficient degree so that airlines, airport, and customers can adjust their habist and process to avoid these delays. 
# MAGIC
# MAGIC ### Process
# MAGIC In this phase, we performed our Exploratory Data Analysis (EDA) on the provided 4 year dataset from 2015-2018, and tested on 2019.
# MAGIC
# MAGIC Through the EDA process we identified various features that acted as inputs to our model experiements. We also created some custom features to add to the input model based on some insights gathered from EDA. 
# MAGIC
# MAGIC After this process, we conducted a walk-forward validation on the 5 year dataset with Random Forest, Gradient Boosting, and 2 versions of Multi-Layer Perceptron models. With the walk-forward validation results, we gathered that there is also likely an important factor of trend across numerical features that could be incorporated in a later iteration.
# MAGIC
# MAGIC Finally, we created baseline hyperparameter tuned models with Random Forest, Gradient Boosting, and 2 versions of Multi-Layer Perceptron models and evaluated those results.
# MAGIC
# MAGIC ### Commentary
# MAGIC We learned from these initial results that the severe imbalance of the input dataset is hindering the output results. The data imbalance of our target variable (whether a flight was delayed or not) is a major obstacle we will have to consider. In this phase we explored basic oversampling which improved the results somewhat, but future models can explore more sophisticated techniques for data augmentation. Furthermore, we will explore more advanced machine learning methods as well as additional custom features in effort to create a more reliable model that can help predict flight delays. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code Notebook References
# MAGIC
# MAGIC Phase 3 EDA (Jason): 
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/4169387258318816
# MAGIC
# MAGIC Extra Credit (2020-2022 Flight Data) (Jason): 
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2926108717729328
# MAGIC
# MAGIC Pagerank (Rodrigo): 
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2926108717723370
# MAGIC
# MAGIC Phase 3 Statistical Tests (Rodrigo): 
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2340268507058200
# MAGIC
# MAGIC Phase 3 Walk-Forward Cross Validation (Rodrigo): 
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2926108717724405 
# MAGIC
# MAGIC Phase 3 RF and GBT (Conner):
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/3544923356436157
# MAGIC
# MAGIC Phase 3 MLPs and Gap Analysis (Kushal): 
# MAGIC * https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/3544923356433479