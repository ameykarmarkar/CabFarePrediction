library(REdaS)
library(fastDummies)
library(MASS)
library(rpart)
# *************************** read database ******************************
datapath <- 'train_cab/train_cab.csv'
data <- read.csv(datapath)

# *************************** Dataset Cleaning **************************
# check for the null values
lapply(data,function(x) { length(which(is.na(x)))})
lapply(data,function(x) { length(which(x ==""))})

row.has.na <- apply(data, 1, function(x){any(is.na(x))})
data.filtered <- data[!row.has.na,]

row.has.null <- apply(data.filtered, 1, function(x){any(x == "")})
data.filtered <- data.filtered[!row.has.null,]
lapply(data.filtered,function(x) { length(which(is.na(x)))})
lapply(data.filtered,function(x) { length(which(x ==""))})

# check for incorrect values
# 1. passenger count
data.filtered = data.filtered[which(data.filtered$passenger_count > 0 & data.filtered$passenger_count < 7 ),]
data.filtered = data.filtered[which(data.filtered$passenger_count != 0.12),]
data.filtered = data.filtered[which(data.filtered$passenger_count != 1.3),]

# 2. latitude, longitude
data.filtered$pickup_longitude = as.numeric(data.filtered$pickup_longitude)
data.filtered$pickup_latitude = as.numeric(data.filtered$pickup_latitude)
data.filtered$dropoff_longitude = as.numeric(data.filtered$dropoff_longitude)
data.filtered$dropoff_latitude = as.numeric(data.filtered$dropoff_latitude)

data.filtered = data.filtered[which(data.filtered$pickup_longitude > -180 & data.filtered$pickup_longitude < 180 ),]
data.filtered = data.filtered[which(data.filtered$dropoff_longitude > -180 & data.filtered$dropoff_longitude < 180 ),]
data.filtered = data.filtered[which(data.filtered$pickup_latitude > -90 & data.filtered$pickup_latitude < 90 ),]
data.filtered = data.filtered[which(data.filtered$dropoff_latitude > -90 & data.filtered$dropoff_latitude < 90 ),]

row.has.na <- apply(data.filtered, 1, function(x){any(is.na(x))})
data.filtered <- data.filtered[!row.has.na,]

data.filtered = data.filtered[which(as.integer(data.filtered$dropoff_latitude) != 0 & as.integer(data.filtered$dropoff_longitude) != 0 ),]

# 3. Fare Amount
#typeof(data.filtered$fare_amount)
data.filtered$fare_amount = as.numeric(as.character(data.filtered$fare_amount))
row.has.na <- apply(data.filtered, 1, function(x){any(is.na(x))})
data.filtered <- data.filtered[!row.has.na,]

data.filtered = data.filtered[which(data.filtered$fare_amount > 0 ),]

outlierDetection <- function(dataColumn) {
  sort(dataColumn)
  lowerupperbound = quantile(dataColumn, c(.25,.75))
  IQR = lowerupperbound[2] - lowerupperbound[1]
  lowerRange = lowerupperbound[1] - (1.5 * IQR)
  upperRange = lowerupperbound[2] + (1.5 * IQR)
  return(list(lowerRange,upperRange)) 
}
lowerupperrange = outlierDetection(data.filtered$fare_amount)
data.filtered = data.filtered[which(data.filtered$fare_amount > lowerupperrange[1] & data.filtered$fare_amount < lowerupperrange[2] ),]

haversine <- function(row){
  lon1 = row[1]
  lat1 = row[2]
  lon2 = row[3]
  lat2 = row[4]
  
  lon1 = deg2rad(lon1)
  lat1 = deg2rad(lat1)
  lon2 = deg2rad(lon2)
  lat2 = deg2rad(lat2)
  
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c =  2 * asin(sqrt(a))
  # Radius of earth in kilometers is 6371
  km = 6371* c
  return(km)
}

data.filtered$distance <- apply(data.filtered[,c(3,4,5,6)], 1, haversine)
lowerupperrange = outlierDetection(data.filtered$distance)
data.filtered = data.filtered[which(data.filtered$distance > lowerupperrange[1] & data.filtered$distance < lowerupperrange[2] ),]

data.filtered$pickupDate = as.Date(as.character(data.filtered$pickup_datetime))
data.filtered$day = as.factor(format(data.filtered$pickupDate,"%u"))# Monday = 1
data.filtered$month = as.factor(format(data.filtered$pickupDate,"%m"))
data.filtered$year = as.factor(format(data.filtered$pickupDate,"%Y"))
pickup_time = strptime(data.filtered$pickup_datetime,"%Y-%m-%d %H:%M:%S")
data.filtered$hour = as.factor(format(pickup_time,"%H"))

hourType <- function(hourTime){
  busyHour <- c(18, 19, 20, 21, 22)
  regularHour <- c(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23)
  if (as.integer(hourTime) %in% busyHour) {
    return('BusyHour')
  } 
  else if (as.integer(hourTime) %in% regularHour) {
    return('RegularHour')
  }
  else{
    return('SlackHour')
  }
}
data.filtered$hourtype <- sapply(data.filtered$hour, hourType)
row.has.na <- apply(data.filtered, 1, function(x){any(is.na(x))})
data.filtered <- data.filtered[!row.has.na,]

data.selected <- subset(data.filtered, select = -c(pickup_datetime, pickupDate, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,month, day, hour) )
data.selected <- dummy_cols(data.selected)
data.selected <- subset(data.selected, select = -c(year, hourtype))

row.has.na <- apply(data.selected, 1, function(x){any(is.na(x))})
data.selected <- data.selected[!row.has.na,]

train_index = sample(1:nrow(data.selected), 0.8 * nrow(data.selected))
train = data.selected[train_index,]
test = data.selected[-train_index,]


# *************** multiple linear regression
model1 <- lm(fare_amount ~., data = train)
summary(model1)

model2 <- stepAIC(model1, direction="both")

model3 <- lm(fare_amount ~ passenger_count + distance + year_2009 + year_2010 + 
               year_2011 + year_2012 + year_2013 + year_2014 + hourtype_BusyHour + 
               hourtype_RegularHour, data = train)
summary(model3)

#year_2014 is having high p-value so dropping feature
model4 <- lm(fare_amount ~ passenger_count + distance + year_2009 + year_2010 + 
               year_2011 + year_2012 + year_2013 + hourtype_BusyHour + 
               hourtype_RegularHour, data = train)
summary(model4)
# All the features are having low p-value so this is our final model
pred <- predict(model4, newdata = test[,2:13])
rmse(pred, test$fare_amount)
# Root mean squared error 2.18 
#*********** Decision Tree Regression ******************************
decisionTree1 <- rpart(
  formula = fare_amount ~ .,
  data    = train,
  method  = "anova"
)
decisionTree1
plotcp(decisionTree1)
## here we can see that after size of tree = 4 we get diminishing returns 
# we will check for max depth from 2 to 10 , as from the above observation we observed optimal value for deth is 4  
hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(2, 10, 1)
)
head(hyper_grid)

models <- list()

for (i in 1:nrow(hyper_grid)) {
  
  # get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # train a model and store in the list
  models[[i]] <- rpart(
    formula = fare_amount ~ .,
    data    = train,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}

# function to get optimal cp
get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# function to get minimum error
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}
library(purrr)
library(dplyr)
library(Metrics)
hyper_grid %>% mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>% arrange(error) %>% top_n(-5, wt = error)

# we will select minsplit = 6 and maxdepth = 5
optimal_tree <- rpart(
  formula = fare_amount ~ .,
  data    = train,
  method  = "anova",
  control = list(minsplit = 6, maxdepth = 5, cp = 0.01)
)

pred <- predict(optimal_tree, newdata = test)
rmse(pred, test$fare_amount)
# Root mean squared error 2.41 

# ***************** Random Forest Regression ********************
library(randomForest)
# for reproduciblity
set.seed(123)

# default RF model
randomForest1 <- randomForest(
  formula = fare_amount ~ .,
  data    = train
)
randomForest1

plot(randomForest1)
# number of trees with lowest MSE
which.min(randomForest1$mse)

# RMSE of this optimal random forest
sqrt(randomForest1$mse[which.min(randomForest1$mse)])

# randomForest
pred_randomForest <- predict(randomForest1, test)
head(pred_randomForest)
rmse(pred_randomForest, test$fare_amount)
#root mean square is 2.19

# ************* predicting fare amount for test dataset
# out of the abobve models random forest and linear regression model has almost same root mean squared error value.
# selecting linear regression model to predict values
datapath <- 'test/test.csv'
testdataoriginal <- read.csv(datapath) 

testdata <- testdataoriginal
lapply(testdata,function(x) { length(which(is.na(x)))})
lapply(testdata,function(x) { length(which(x ==""))})
# no null or empty value

# check for incorrect values
# 1. passenger count
testdata = testdata[which(testdata$passenger_count > 0 & testdata$passenger_count < 7 ),]
print(unique(testdata$passenger_count))
#all values are correct

# 2. latitude, longitude
testdata$pickup_longitude = as.numeric(testdata$pickup_longitude)
testdata$pickup_latitude = as.numeric(testdata$pickup_latitude)
testdata$dropoff_longitude = as.numeric(testdata$dropoff_longitude)
testdata$dropoff_latitude = as.numeric(testdata$dropoff_latitude)

testdata = testdata[which(testdata$pickup_longitude > -180 & testdata$pickup_longitude < 180 ),]
testdata = testdata[which(testdata$dropoff_longitude > -180 & testdata$dropoff_longitude < 180 ),]
testdata = testdata[which(testdata$pickup_latitude > -90 & testdata$pickup_latitude < 90 ),]
testdata = testdata[which(testdata$dropoff_latitude > -90 & testdata$dropoff_latitude < 90 ),]
# data is correct

# finding distace of trip

testdata$distance <- apply(testdata[,c(2,3,4,5)], 1, haversine)
testdata$pickupDate = as.Date(as.character(testdata$pickup_datetime))
testdata$day = as.factor(format(testdata$pickupDate,"%u"))# Monday = 1
testdata$month = as.factor(format(testdata$pickupDate,"%m"))
testdata$year = as.factor(format(testdata$pickupDate,"%Y"))
pickup_time = strptime(testdata$pickup_datetime,"%Y-%m-%d %H:%M:%S")
testdata$hour = as.factor(format(pickup_time,"%H"))

testdata$hourtype <- sapply(testdata$hour, hourType)
#row.has.na <- apply(testdata, 1, function(x){any(is.na(x))})
#testdata <- testdata[!row.has.na,]

testdata <- subset(testdata, select = -c(pickup_datetime, pickupDate, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,month, day, hour) )
testdata <- dummy_cols(testdata)
testdata <- subset(testdata, select = -c(year, hourtype))
pred <- predict(model4, newdata = testdata)
print(head(pred))
testdataoriginal$fare_amount = pred
write.csv(testdataoriginal, "predicted_cabfare_data.csv")
