# Human Activity Prediction
========================================================

## Introduction

Use practical machine learning to predict human activity on HAR dataset. The dataset has been obtained from http://groupware.les.inf.puc-rio.br/har. 


## Pre-Processing data
The labelled data consists of a CSV file which can be loaded in R. Pre-processing includes all the steps to be performed before we actually start applying algorithms to learn from data. It includes generating train/test sets, cleanup, and normalization.

### Generate train / test sets
We generate train and test sets out of the original labelled data for evaluating our model.


```r
library(caret)

# Load data, treat empty and #DIV/0! strings as NA -
dataCSV <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))

# Set seed for reproducability
set.seed(13309)

# Partition to keep 80% data for training, and 20% for
# testing/cross-validation
inTrain <- createDataPartition(y = dataCSV$classe, p = 0.8, list = FALSE)
trainCSV <- dataCSV[inTrain, ]
testCSV <- dataCSV[-inTrain, ]
```


So data is divided from total 19622 to training set with 15699 rows and test set with 3923 rows. We will not touch the test set, treat it as a blind data, and perform the rest of pre-processing by exploring training data. 

### Cleaning data
Training set has 160 columns. We reduce them as follows -

*Remove un-predictive columns*
The first column 'X' is just an index column, so we remove it.


```r
trainClean1 <- trainCSV[, -1]
```


*Remove empty columns*
We observe that a lot of columns consist only of NA in data -


```r
plot(colSums(is.na(trainClean1)), ylab = "Count of NAs", main = "NA frequency per column in training data")
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 



```r
# 95% of row count -
nrow_95_pc <- (9.5 * nrow(trainClean1))/10

# Columns with more than 95% rows as NA
columns_with_95_pc_nas <- which(colSums(is.na(trainClean1)) > nrow_95_pc)
```


There are 100 columns with more than 95% rows as NA. So we remove those columns as part of cleanup -


```r
trainClean2 <- trainClean1[, -columns_with_95_pc_nas]
```


*Reduce correlated numeric variables*
We identify the pair of numeric variables with high correlation


```r
# Columns of type numeric -
numeric_columns <- which(sapply(trainClean2, class) != "factor")

count_numeric <- length(numeric_columns)

# Find index pairs with absolute correlation > 0.8
all_correlated_flat_pairs <- which(abs(cor(trainClean2[numeric_columns], trainClean2[numeric_columns])) > 
    0.8)

# Remove the pairs like (1,1) (2,2) etc.
same_flat_pairs <- sapply(c(0, seq_along(numeric_columns)), function(i) {
    count_numeric * i + i + 1
})
correlated_flat_pairs <- setdiff(all_correlated_flat_pairs, same_flat_pairs)

# This is a list of unique correlated index pairs,
correlated_index_pairs <- unique(lapply(correlated_flat_pairs, function(i) {
    index1 <- floor(i/count_numeric) + 1
    index2 <- i%%count_numeric
    if (index2 < index1) {
        temp <- index2
        index2 <- index1
        index1 <- temp
    }
    c(numeric_columns[index1], numeric_columns[index2])
}))
```


Let's take the first 2 pairs with high correlation -

```r
correlated_index_pairs[1:2]
```

```
## [[1]]
## roll_belt  yaw_belt 
##         7         9 
## 
## [[2]]
##        roll_belt total_accel_belt 
##                7               10
```


*Plotting corrrelated columns*
As we see, roll_belt (index 7), yaw_belt(index 9) and total_accel_belt(index 10) have high mutual corrrelation. Here is a plot depicting the same, darker shade implies higher correlation.


```r
library(corrgram)
```

```
## Loading required package: seriation
```

```r

# Plot correlation for columns : 7,8,9,10
corrgram(trainClean2[, 7:10], panel = panel.shade)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8.png) 


So we remove the first index of each such pairs.

```r
remove_numeric_columns <- unique(sapply(correlated_index_pairs, function(elem) {
    elem[[1]]
}))

trainCleanFinal <- trainClean2[, -remove_numeric_columns]
```


To put things in perspective, we notice that shades are not that dark now -


```r
# Plot correlation for columns : 7,8,9,10
corrgram(trainCleanFinal[, 7:10], panel = panel.shade)
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10.png) 



## Modelling
We train a random forest algorithm over the cleaned training set.


```r
# Use parallelization across cores for quicker computation
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Train a random forest (rf) model with standardized training set and using
# cross validation
trainCleanFinal$classe <- factor(trainCleanFinal$classe)
rf_model <- train(classe ~ ., data = trainCleanFinal, method = "rf", preProcess = c("center", 
    "scale"), trControl = trainControl(method = "cv"))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
print(rf_model)
```



```r
print(rf_model)
```

```
## Random Forest 
## 
## 15699 samples
##    45 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 14129, 14129, 14129, 14129, 14131, 14128, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.004        0.005   
##   30    1         1      0.001        0.001   
##   70    1         1      0.001        0.001   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 34.
```


## Cross validation
We now analyze our model by computing in-sample error (i.e. error rate on training set) and out of sample error (i.e. error rate on testing set).

### In sample error

```r
train_total_rows <- nrow(trainCleanFinal)

train_prediction <- predict(rf_model, trainCleanFinal)

train_correctly_predicted_rows <- length(which(train_prediction == trainCleanFinal$classe))
in_sample_error <- 100 * (1 - (1 * train_correctly_predicted_rows/train_total_rows))

table(train_prediction, trainCleanFinal$classe)
```

The in-sample error rate for the learnt model is 0%.

### Out of sample error

```r
testClean1 <- testCSV[, -1]
testClean2 <- testClean1[, -columns_with_95_pc_nas]
testCleanFinal <- testClean2[, -remove_numeric_columns]
test_total_rows <- nrow(testCleanFinal)

test_prediction <- predict(rf_model, testCleanFinal)

test_correctly_predicted_rows <- length(which(test_prediction == testCleanFinal$classe))
out_of_sample_error <- 100 * (1 - (1 * test_correctly_predicted_rows/test_total_rows))

table(test_prediction, testCleanFinal$classe)
```

The out-of-sample error rate for the learnt model is 0.0765%.
