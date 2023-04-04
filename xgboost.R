library(xgboost)
library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

library(caret)

load("class_data.Rdata")
set.seed(1)
trainIndex =sample(1:nrow(x),250)
x_train <- x[trainIndex, ]
x_train <- scale(x_train)
y_train <- y[trainIndex]
x_test <- x[-trainIndex, ]
y_test <- y[-trainIndex]

# x_train=data.frame(x_train, y_train)
# x_test=data.frame(x_test, y_test)
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
# Set number of boosting rounds
nrounds <- 100
early_stopping_rounds <- 10

# Set parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "error"
)
# # Use xgb.cv for cross-validation
# cv_results <- xgb.cv(params = params, data = dtrain, nrounds = nrounds,
#                      early_stopping_rounds = early_stopping_rounds,
#                      maximize = FALSE, nfold = 10, seed = 1)

# Get the best number of boosting rounds based on cross-validation results
# best_nrounds <- which.min(cv_results$evaluation_log$test_error_mean)
# Train xgboost model
model <- xgb.train(data = dtrain, nrounds = 100)
# 
# # Get feature importance scores
importance_scores <- xgb.importance(model = model)
xgboostpredict = predict(model,data.matrix(y_test))
xgboostpredict[xgboostpredict < 0.5] = 0
xgboostpredict[xgboostpredict >= 0.5] = 1

matrix.train <- table(xgboostpredict, x_test$y_test)
accuracy = mean(xgboostpredict == x_test$y_test)

# print(importance_scores$Feature)
# 
# features = importance_scores$Feature
# 
# 
# 
# x_train = x_train[, features]
# boost.boston=gbm(y_train~.,data=x_train,distribution="bernoulli",n.trees=500,shrinkage=0.0001,interaction.depth=9,cv.fold=10)
# n.trees.cv=gbm.perf(boost.boston, method = "cv")
# gbmpred=predict.gbm(boost.boston,newdata=x_test,n.trees=n.trees.cv, type = "response")



