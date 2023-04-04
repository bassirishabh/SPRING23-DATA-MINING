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
# glimpse(x)
# table(y)

set.seed(123)

hyper_grid <- expand.grid(
  interaction.depth = c(5, 9, 15, 25),
  n.trees = c(500, 1000, 2500, 4000),
  shrinkage = c(0.1, 0.01, 0.001, 0.0001),
  n.minobsinnode = 10
)

train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats=2,
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

library(dplyr)

