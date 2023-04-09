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
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = "final")

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

# Set up cross-validation and hyperparameter tuning
xgb_grid <- expand.grid(
  nrounds = c(100,200),
  max_depth = c(5,9, 15),
  eta = c(0.1, 0.01, 0.001),
  gamma = c(0,1),
  colsample_bytree = c(0.3, 0.5, 0.7, 0.9),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.1, 0.3, 0.5, 0.7, 0.9)
)


start = Sys.time()

# train xgboost model with feature selection
xgb_model <- train(x = X_train,
                   y = y_train,
                   method = "xgbTree",
                   trControl = ctrl,
                   verbose = TRUE,
                   tuneGrid = xgb_grid,
                   metric = "ROC",
                   verbosity = 0
                   )
end = Sys.time()

# # Save the model
end_date = format(end, "%Y_%m_%d")
file_name = paste0("xgbTree_model_",end_date,".RData")
#
save(xgb_model, file = file_name)

# load(file_name)

# get feature importance
importance <- xgb.importance(dimnames(X_train)[[2]], model = xgb_model$finalModel)

# plot feature importance
xgb.plot.importance(importance)

# make predictions on test set
xgb_predictions <- predict(xgb_model, newdata = X_test, iteration_range = c(1, 50))

# evaluate performance
confusionMatrix(xgb_predictions, y_test)

library(ROCR)
PredBoosting <- predict(xgb_model, X_test,type = "prob")
prediction <- prediction(PredBoosting[2],y_test)
performance <- performance(prediction, "tpr","fpr")
# plotting ROC curve
plot(performance,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")

# area under curve
aucXGBoost <- performance(prediction, measure = "auc")
aucXGBoost <- aucXGBoost@y.values[[1]]
aucXGBoost
