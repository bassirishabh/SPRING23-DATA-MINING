library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

library(tidyverse)
# package to compute
# cross - validation methods
library(caret)

load("class_data.Rdata")
y <-as.factor(y)
levels(y) <-make.names(y)
glimpse(x)
table(y)

set.seed(123)

hyper_grid <- expand.grid(
  interaction.depth = c(5, 9, 15, 25),
  n.trees = c(500, 1000),
  shrinkage = 0.0001,
  n.minobsinnode = 10
)

train_control <- trainControl(method = "cv",
                              number = 10,
                              savePredictions = "final",
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              search = "grid"
                              )

model <- train(x = x, y = y,
               metric = "ROC",
               trControl = train_control,
               tuneGrid = hyper_grid,
               method = "gbm")

print(model$results)

# Calculate the misclassification rate
cm <- confusionMatrix(model$pred$pred, model$pred$obs)
misclassification_rate <- cm$overall[1]   # extract the accuracy metric

# Print the misclassification rate
cat("Misclassification rate:", misclassification_rate, "\n")

# # Plot confusion matrices and ROC curves for each fold
# for (i in 1:length(model$pred)) {
#   cm <- confusionMatrix(model$pred[[i]]$pred, model$pred[[i]]$obs)
#   plot(cm$table, main = paste("Fold", i), col = c("#0072B2", "#D55E00"))
#   roc <- roc(model$pred[[i]]$obs, model$pred[[i]]$R1)
#   plot(roc, main = paste("Fold", i), col = "#0072B2")
# }
# 
# boost.boston=gbm(y~.,data=x,distribution="multinomial",n.trees=500,shrinkage=0.0001,interaction.depth=5,cv.fold=10)
# n.trees.cv=gbm.perf(boost.boston, method = "cv")
# gbmpred=predict.gbm(boost.boston,newdata=x_test,n.trees=n.trees.cv, type = "response")
# class_names = colnames(gbmpred)[apply(gbmpred, 1, which.max)]
# boost_err = 1-mean(class_names == x_test$y_test_up)

