# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:17:05 2020

@author: ameyk
"""


# **************************************** Import Libraries *****************************************************************
import pandas as pd
#from math import radians, cos, sin, asin, sqrt
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics import r2_score
import xgboost as xgb
# **************************************** Read Data *************************************************************************

cabFareData = pd.read_csv("train_cab/train_cab.csv")
print(cabFareData.head())

# *************************************** Data Cleaning **********************************************************************
## print size of data
print(cabFareData.shape)
# 16067 trip data 

## check NA
print(cabFareData.isna().sum())
print((cabFareData.isna().sum()/cabFareData.shape[0])*100)
## drop na value containing rows
cabFareData = cabFareData[cabFareData['fare_amount'].isna() == False]
cabFareData = cabFareData[cabFareData['passenger_count'].isna() == False]
print(cabFareData.isna().sum())

## check for incorrect values
# 1. passenger_count
cabFareData = cabFareData[(cabFareData['passenger_count'] > 0) & (cabFareData['passenger_count'] < 7)]
#cabFareData unique values
print((cabFareData.passenger_count.unique()))
#passeneger count can not be in decimal
# removing passenger count with value 1.3 and 0.12
cabFareData = cabFareData[(cabFareData['passenger_count'] != 0.12)]
cabFareData = cabFareData[(cabFareData['passenger_count'] != 1.3)]

# 2. Latitude, Longitude value for pickup and dropoff location
cabFareData['pickup_longitude'] = pd.to_numeric(cabFareData['pickup_longitude'], errors = "coerce")
cabFareData = cabFareData[cabFareData['pickup_longitude'].isna() == False]
cabFareData = cabFareData[(cabFareData['pickup_longitude'] > -180) & (cabFareData['pickup_longitude'] < 180)]

cabFareData['pickup_latitude'] = pd.to_numeric(cabFareData['pickup_latitude'], errors = "coerce")
cabFareData = cabFareData[cabFareData['pickup_latitude'].isna() == False]
cabFareData = cabFareData[(cabFareData['pickup_latitude'] > -90) & (cabFareData['pickup_latitude'] < 90)]

cabFareData = cabFareData[(cabFareData['pickup_latitude'] == 0) & (cabFareData['pickup_longitude'] == 0) == False]

cabFareData['dropoff_longitude'] = pd.to_numeric(cabFareData['dropoff_longitude'], errors = "coerce")
cabFareData = cabFareData[cabFareData['dropoff_longitude'].isna() == False]
cabFareData = cabFareData[(cabFareData['dropoff_longitude'] > -180) & (cabFareData['dropoff_longitude'] < 180)]

cabFareData['dropoff_latitude'] = pd.to_numeric(cabFareData['dropoff_latitude'], errors = "coerce")
cabFareData = cabFareData[cabFareData['dropoff_latitude'].isna() == False]
cabFareData = cabFareData[(cabFareData['dropoff_latitude'] > -90) & (cabFareData['dropoff_latitude'] < 90)]

cabFareData = cabFareData[(cabFareData['dropoff_latitude'] == 0) & (cabFareData['dropoff_longitude'] == 0) == False]
print(cabFareData.shape)

cabFareData['pickup_datetime'] = pd.to_datetime(cabFareData['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', errors = "coerce")
cabFareData = cabFareData[cabFareData['pickup_datetime'].isna() == False]
print(cabFareData.shape)

cabFareData['fare_amount'] = pd.to_numeric(cabFareData['fare_amount'], errors = "coerce")
cabFareData = cabFareData[cabFareData['fare_amount'].isna() == False]
cabFareData = cabFareData[cabFareData['fare_amount'] > 0]
print(cabFareData.shape)

cabFareData.to_csv("CabFareDataWithOutlier.csv")

sns.set(style="whitegrid")
ax = sns.boxplot(x=cabFareData["fare_amount"])

## Outlier Detection
def outlierDetection(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

lowerbound,upperbound = outlierDetection(cabFareData.fare_amount)
print(lowerbound,upperbound)
cabFareData = cabFareData[(cabFareData.fare_amount >= lowerbound) & (cabFareData.fare_amount <= upperbound)]

def haversine(row):
    lon1=row[0]
    lat1=row[1]
    lon2=row[2]
    lat2=row[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min 


# In[46]:


cabFareData['distance'] = cabFareData[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
ax = sns.boxplot(x=cabFareData["distance"])
lowerbound,upperbound = outlierDetection(cabFareData.distance)
print(lowerbound,upperbound)
cabFareData = cabFareData[(cabFareData.distance >= lowerbound) & (cabFareData.distance <= upperbound)]


cabFareData['pickup_datetime'] = pd.to_datetime(cabFareData['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', errors = "coerce")
cabFareData = cabFareData[cabFareData['pickup_datetime'].isna() == False]
print(cabFareData.shape)
cabFareData['year'] = cabFareData['pickup_datetime'].dt.year
cabFareData['Month'] = cabFareData['pickup_datetime'].dt.month
cabFareData['Date'] = cabFareData['pickup_datetime'].dt.day
cabFareData['Day'] = cabFareData['pickup_datetime'].dt.dayofweek
cabFareData['Hour'] = cabFareData['pickup_datetime'].dt.hour
cabFareData['Minute'] = cabFareData['pickup_datetime'].dt.minute



cabFareData['pickup_latitude'] = cabFareData['pickup_latitude'].astype('float')
cabFareData['pickup_longitude'] = cabFareData['pickup_longitude'].astype('float')
cabFareData['dropoff_latitude'] = cabFareData['dropoff_latitude'].astype('float')
cabFareData['dropoff_longitude'] = cabFareData['dropoff_longitude'].astype('float')

cabFareData.to_csv("CabfareDataAnalyzeBusyHours.csv")

#based on the visualization from tableau 
#18, 19, 20, 21, 22 are the busy hours
#7,8,9,10,11,12,13,14.,15,17,23 - regular hours
#0,1,2,3,4,5,6 - Slack hours
def assignHourCategory(hourTime):
    if hourTime in [18, 19, 20, 21, 22]:
        return "BusyHour"
    elif hourTime in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23]:
        return "RegularHour"
    else:
        return "SlackHour"

cabFareData['hour_type'] = cabFareData['Hour'].apply(assignHourCategory)
cabFareData.to_csv("CabfareDataAnalyzedBusyHours.csv")

# dropping variables which are used to derive new variables
selectedCabFareData = cabFareData.drop(['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude', 'Month','Day','Date','Minute','Hour'], axis = 1)

#create dummies for categorical variables year and hour_type
year_dummies = pd.get_dummies(selectedCabFareData['year'])
selectedCabFareData = pd.concat([selectedCabFareData, year_dummies], axis = 1)

hour_type_dummies = pd.get_dummies(selectedCabFareData['hour_type'])
selectedCabFareData = pd.concat([selectedCabFareData, hour_type_dummies], axis = 1)

selectedCabFareData = selectedCabFareData.drop(['year','hour_type'], axis = 1)


X = selectedCabFareData.drop(['fare_amount'], axis = 1)
y = selectedCabFareData['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# ************************************* Mulitple Linear Regression ************************************************

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient']) 
print(coeff_df)


y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R square value is: ", r2_score(y_test,y_pred))
print(selectedCabFareData.columns)
#RMSE 2.07
# R square value 0.68
# ********************************************* Decision Tree Regressor ***********************************************
max_depth_values = [2,3,4,5,6,7, 10, 15]
rmse_values = {}
r_square_values = {}
for max_depth_value in max_depth_values: 
    dt = DecisionTreeRegressor(max_depth = max_depth_value, min_samples_split = 10)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, dt_pred))  
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, dt_pred))  
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dt_pred)))
    #print("R square value is: ", r2_score(y_test,dt_pred))
    rmse_values[max_depth_value] = np.sqrt(metrics.mean_squared_error(y_test, dt_pred))
    r_square_values[max_depth_value] = r2_score(y_test,dt_pred)
print(min(rmse_values, key=rmse_values.get))
min_rmse_depth_value = min(rmse_values, key=rmse_values.get)
print(min_rmse_depth_value)
dt_model = DecisionTreeRegressor(max_depth = min_rmse_depth_value, min_samples_split = 10)
dt_model.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dt_pred)))
print("R square value is: ", r2_score(y_test,dt_pred))
#RMSE 2.42
#R square .56
    
# ************************************************* Random Forest Prediction *****************************************
estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
rmse_values = {}
r_square_values = {}

for estmator in estimators:
    rf = RandomForestRegressor(n_estimators = estmator, random_state = 43,n_jobs=-1)
    rf.fit(X_train,y_train)
    rf_pred= rf.predict(X_test)
    rmse_values[estmator] = np.sqrt(metrics.mean_squared_error(y_test, rf_pred))
    r_square_values[estmator] = r2_score(y_test,rf_pred)
    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rf_pred))  
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, rf_pred))  
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
    #print("R square value is: ", r2_score(y_test,rf_pred))
estimator_value = min(rmse_values, key=rmse_values.get)
rf = RandomForestRegressor(n_estimators = estimator_value, random_state = 43,n_jobs=-1)
rf.fit(X_train,y_train)
rf_pred= rf.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
print("R square value is: ", r2_score(y_test,rf_pred))
#RMSE 2.24
#R square .62

# ************** predicting fare amount for test dataset *****************************
# Root mean square value of linear regression model is lowest and R square value is highest so selecting linear regression model
cabFareTestOriginalData = pd.read_csv("test/test.csv")
print(cabFareTestOriginalData.head())
cabFareTestData = cabFareTestOriginalData.copy()
cabFareTestData['pickup_longitude'] = pd.to_numeric(cabFareTestData['pickup_longitude'], errors = "coerce")
cabFareTestData['pickup_latitude'] = pd.to_numeric(cabFareTestData['pickup_latitude'], errors = "coerce")
cabFareTestData['dropoff_longitude'] = pd.to_numeric(cabFareTestData['dropoff_longitude'], errors = "coerce")
cabFareTestData['dropoff_latitude'] = pd.to_numeric(cabFareTestData['dropoff_latitude'], errors = "coerce")
cabFareTestData['pickup_datetime'] = pd.to_datetime(cabFareTestData['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC', errors = "coerce")

cabFareTestData['distance'] = cabFareTestData[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)

cabFareTestData['year'] = cabFareTestData['pickup_datetime'].dt.year
cabFareTestData['Month'] = cabFareTestData['pickup_datetime'].dt.month
cabFareTestData['Date'] = cabFareTestData['pickup_datetime'].dt.day
cabFareTestData['Day'] = cabFareTestData['pickup_datetime'].dt.dayofweek
cabFareTestData['Hour'] = cabFareTestData['pickup_datetime'].dt.hour
cabFareTestData['Minute'] = cabFareTestData['pickup_datetime'].dt.minute



cabFareTestData['pickup_latitude'] = cabFareTestData['pickup_latitude'].astype('float')
cabFareTestData['pickup_longitude'] = cabFareTestData['pickup_longitude'].astype('float')
cabFareTestData['dropoff_latitude'] = cabFareTestData['dropoff_latitude'].astype('float')
cabFareTestData['dropoff_longitude'] = cabFareTestData['dropoff_longitude'].astype('float')

cabFareTestData['hour_type'] = cabFareTestData['Hour'].apply(assignHourCategory)


# dropping variables which are used to derive new variables
cabFareTestData = cabFareTestData.drop(['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude', 'Month','Day','Date','Minute','Hour'], axis = 1)

#create dummies for categorical variables year and hour_type
year_dummies = pd.get_dummies(cabFareTestData['year'])
cabFareTestData = pd.concat([cabFareTestData, year_dummies], axis = 1)

hour_type_dummies = pd.get_dummies(cabFareTestData['hour_type'])
cabFareTestData = pd.concat([cabFareTestData, hour_type_dummies], axis = 1)

cabFareTestData = cabFareTestData.drop(['year','hour_type'], axis = 1)
y_pred = regressor.predict(cabFareTestData)
print(y_pred[0:10])
cabFareTestOriginalData['fare_amount'] = y_pred
cabFareTestOriginalData.to_csv("predicted_cabfare_data_python.csv")















