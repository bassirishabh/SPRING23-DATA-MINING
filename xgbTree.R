library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

library(xgboost)
library(caret)

# load data
load("class_data.RData")

# split data into training and testing sets
set.seed(123)
y <- as.factor(y)
levels(y) <-make.names(y)
train <- sample(1:nrow(x), nrow(x) * 0.7)
X_train <- x[train, ]
y_train <- y[train]
X_test <- x[-train, ]
y_test <- y[-train]

# define cross-validation method
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

# Set up cross-validation and hyperparameter tuning
# xgb_grid <- expand.grid(
#   nrounds = c(100,200),
#   max_depth = c(5,9, 15),
#   eta = c(0.1, 0.01, 0.001),
#   gamma = c(0,1),
#   colsample_bytree = c(0.8, 0.9),
#   min_child_weight = c(1,5, 10),
#   subsample = c(0.1, 0.3, 0.5, 0.7, 0.9)
# )

xgb_grid <- expand.grid(
  nrounds = c(100,200),
  max_depth = c(5,9),
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 0.8,
  rate_drop =  0,
  skip_drop = 0.5,
  min_child_weight = 1,
  subsample = 0.8
)

# train xgboost model with feature selection
xgb_model <- train(x = X_train, 
                   y = y_train, 
                   method = "xgbDART",
                   trControl = ctrl, 
                   verbose = FALSE,
                   tuneGrid = xgb_grid,
                   metric = "ROC",
                   verbosity = 0
                   )

# get feature importance
importance <- xgb.importance(dimnames(X_train)[[2]], model = xgb_model$finalModel)

# plot feature importance
xgb.plot.importance(importance)

# make predictions on test set
predictions <- predict(xgb_model, newdata = X_test, iteration_range = c(1, 50))

# evaluate performance
confusionMatrix(predictions, y_test)
