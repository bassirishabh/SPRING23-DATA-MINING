library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

# Load the required packages
library(caret)
library(randomForest)

# Load the data
load("class_data.Rdata")
set.seed(123)
y <-as.factor(y)
levels(y) <-make.names(y)

# Define the number of folds for cross-validation
num_folds <- 10


train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats=2,
                              savePredictions = "final",
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              search = "grid"
)


hyper_grid <- expand.grid(
  mtry = c(50, 100, 150, 22)
)

# Define the ordinalRF model
model <- train(x = x, y = y,
               metric = "Accuracy",
               trControl = train_control,
               tuneGrid = hyper_grid,
               method = "rf")

# Print the model performance
print(model)

# Plot the variable importance
varImpPlot(model$finalModel)

# Plot the confusion matrix and ROC curve for each fold
for (i in 1:num_folds) {
  cm <- confusionMatrix(model$pred[[i]]$pred, model$pred[[i]]$obs)
  plot(cm$table, main = paste("Fold", i), col = c("#0072B2", "#D55E00"))
  roc <- roc(model$pred[[i]]$obs, model$pred[[i]]$Yhat[, 1])
  plot(roc, main = paste("Fold", i), col = "#0072B2")
}
