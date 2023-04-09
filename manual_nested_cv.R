library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

# Load data
library(MASS)
library(gbm)
library(caret)
load("class_data.RData")

# Creating Folds
folds <- cut(seq(1,nrow(x)),breaks=10,labels=FALSE)

best_list = vector(mode='list', length=10)
accuracy = vector(mode='list', length=10)
for(i in 1:10){
  cat("Loop:", i, "\n")
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  x_test <- x[testIndexes, ]
  x_train <- x[-testIndexes, ]
  y_test <- y[testIndexes ]
  y_train <- y[-testIndexes ]
  #Use the test and train data partitions however you desire...
  

  #CV Tuning
  hyper_grid <- expand.grid(
    interaction.depth = c(5),
    n.trees = c(500, 1000),
    shrinkage = 0.0001,
    n.minobsinnode = 10
  )
  
  model <- train(x = x_train, y = y_train,
                 metric = "accuracy",
                 tuneGrid = hyper_grid,
                 method = "gbm", verbose = FALSE)
  
  best_params = model$bestTune
  best_list[[i]] = best_params
  
  x_train=data.frame(x_train, y_train)
  gbm.model=gbm(y_train~., 
                data=x_train,
                distribution="bernoulli",
                n.trees=best_params$n.trees,
                shrinkage=best_params$shrinkage,
                interaction.depth=best_params$interaction.depth, 
                cv.fold=10,
                verbose = FALSE)
  
  n.trees.cv=gbm.perf(gbm.model, method = "cv")
  gbm.pred=predict.gbm(gbm.model,newdata=x_test,n.trees=n.trees.cv, type = "response")
  gbm.pred[gbm.pred <= 0.5] = 0
  gbm.pred[gbm.pred > 0.5] = 1
  matrix.train <- table(gbm.pred, y_test)
  classification_accuracy = mean(gbm.pred == y_test)
  accuracy[[i]] = classification_accuracy
  cat("Accuracy rate:", classification_accuracy, "\n")
}
