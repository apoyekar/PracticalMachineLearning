# Practical Machine Learning Project
Anand P  
Saturday, Saturday 22, 2015  

In this report, we will build machine learning algorithm to predict activity quality (activity class) from activity monitors. We will use Weight Lifting Excerse dataset from the website http://groupware.les.inf.puc-rio.br/har. It consists of data from accelerometers on the belt, forearm, arm and dumbell of 6 participants which were asked to perform barbell lifts correctly and incorrectly in 5 different ways. These activities are classefied into A (correct), B, C, D, E. We will then use Cross Validation to validate our model and use the model to predict a set of 20 observations.

## Summary

We cleaned the dataset by removing few variables and then used Random Forest to build model with 100 trees. The model gives the accuracy of 0.9942 when cross validated on the test data which was not used during the training. Then we used selected model to predict 20 observations.

## Data

Weight Lifting Exercises Data Set: is to investigate "how well" this activity was performed by 6 male participants using a dumbbell (1.25kg).

Four sensors were used on: forearm, arm, lumbar belt and dumbbell.

* Data per sensor:
* Position: roll, pitch, yaw
* Acceleration - 3 axis (x, y, z)
* Gyroscope - 3 axis (x, y, z)
* Magnetometer - 3 axis (x, y, z)

Total Data per sensor: 3 X 4 = 12


### Loading the data:


```r
setwd("D:/projects/OnlineCourses/08 Practical Machine Learning/project")
data <- read.csv("pml-training.csv",na.strings = c("", " ","NA"))
```

### Cleaning the Data

There are total 160 variables in the dataset.

We will remove variables that were created for calculation purposes. These variables were created for groups of measurements defined by variable "new_window". 
So we will remove variables starting with 
* kurtosis_ 
* skewness_
* min_
* max_
* amplitude_
* var_
* avg_
* stddev_

we will treat each measurement as independent observation ignoring following variables (first 7 variables):
* first variable X which identifies row
* raw_timestamp_part_1
* raw_timestamp_part_2
* cvtd_timestamp
* user_name which identifies user
* new_window
* num_window

We are now left with 53 variables. The data doesn't have any NAs.


```r
f <- grepl("^kurtosis_|^skewness_|min_|max_|amplitude_|var_|avg_|stddev_", names(data));
f[1:7] <- TRUE # remove first 7 variables
data <- data[,!f];
str(data);
```

```
## 'data.frame':	19622 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
any(is.na(data))
```

```
## [1] FALSE
```

## Building Model

### Creating Training and Test Datasets

Let's first create training and test datasets

```r
library(caret);
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(113);
inTrain <- createDataPartition(y=data$classe, p=0.70,list=FALSE)
training <- data[inTrain,];
testing <- data[-inTrain,];
```

We will use directly use randomForest function to build our model with ntree as 100 since it takes forever if we try to build random forest model using train function in caret package.

```r
library(randomForest)
rf <- randomForest(classe~.,data=training,PROXIMITY=TRUE,ntree=100);
rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, PROXIMITY = TRUE,      ntree = 100) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.63%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    3    0    0    0 0.0007680492
## B   14 2637    7    0    0 0.0079006772
## C    0   18 2372    6    0 0.0100166945
## D    0    0   27 2224    1 0.0124333925
## E    0    0    1    9 2515 0.0039603960
```

### Importance of variables 
Let's see most important of variables in our model:


```r
library(randomForest)
varImpPlot(rf,n.var=5)
```

![](index_files/figure-html/unnamed-chunk-5-1.png) 

Variables roll_belt, yaw_belt, magnet_dumbbell_z, pitch_forearm, magnet_dumbbell_y, pitch_belt seems to be more important.

# Cross Validation

Let's do cross validation the test data. Although Random Forest internally does CV while selecting model, we will test our model on the test data which was not used during training.

Impurity in classification models is measured with misclassification error, Gini index and 
Deviance or Information Gain.

Model predicted class A in all 1672 cases out of 1674 and predicted class B in 1131 cases out of 1139 cases with class B.

Misclassfication Error for class B = 8/1131 = 0.00702

Overall Accuracy is 0.9942 and Out of Sample error rate = 1 - 0.9942 = 0.0058.


```r
library(randomForest)
pred <- predict(rf,testing[,-53])
confusionMatrix(pred,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    2 1131    4    0    0
##          C    0    5 1020   16    0
##          D    0    0    2  947    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9930   0.9942   0.9824   0.9991
## Specificity            0.9993   0.9987   0.9957   0.9994   0.9998
## Pos Pred Value         0.9982   0.9947   0.9798   0.9968   0.9991
## Neg Pred Value         0.9995   0.9983   0.9988   0.9966   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1922   0.1733   0.1609   0.1837
## Detection Prevalence   0.2846   0.1932   0.1769   0.1614   0.1839
## Balanced Accuracy      0.9990   0.9959   0.9949   0.9909   0.9994
```

There is A reduction of error rate as no of trees was increased from 20 to 100. 

## Predicting 20 test cases

We will now use this model to 20 test cases. First we will clean the test dataset and remove the same variables that we removed from our training dataset. 

```r
t <- read.csv("pml-testing.csv",na.strings = c("", " ","NA"));
t <- t[,!f];
pred <- predict(rf,newdata=t[,-53])
pred
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

We will create output file with prediction for each test case using the following function.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred)
```





