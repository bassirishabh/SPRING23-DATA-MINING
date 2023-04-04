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
y <- as.factor(y)
levels(y) <- make.names(y)
glimpse(x)
table(y)

set.seed(123)

hyper_grid <- expand.grid(
  interaction.depth = c(5, 9, 15, 25),
  n.trees = c(500, 1000),
  shrinkage = 0.0001,
  n.minobsinnode = 10
)

# Define the outer cross-validation folds
outer_folds <- createFolds(y, k = 5, returnTrain = TRUE)

# Initialize a list to store the results
result_list <- list()

# Loop through the outer folds
for (i in seq_along(outer_folds)) {
  
  # Get the training and test data for this fold
  train_indices <- unlist(outer_folds[-i])
  test_indices <- outer_folds[[i]]
  x_train <- x[train_indices, ]
  y_train <- y[train_indices]
  x_test <- x[test_indices, ]
  y_test <- y[test_indices]
  
  # Define the inner cross-validation folds
  inner_folds <- createFolds(y_train, k = 10, returnTrain = TRUE)
  
  # Initialize a list to store the inner results
  inner_result_list <- list()
  
  # Loop through the inner folds
  for (j in seq_along(inner_folds)) {
    
    # Get the training and validation data for this fold
    train_indices <- unlist(inner_folds[-j])
    val_indices <- inner_folds[[j]]
    x_train_inner <- x_train[train_indices, ]
    y_train_inner <- y_train[train_indices]
    x_val <- x_train[val_indices, ]
    y_val <- y_train[val_indices]
    
    # Train a gbm model using this set of hyperparameters
    model <- gbm(x = x_train_inner, y = y_train_inner,
                 distribution = "bernoulli",
                 interaction.depth = hyper_grid$interaction.depth,
                 n.trees = hyper_grid$n.trees,
                 shrinkage = hyper_grid$shrinkage,
                 n.minobsinnode = hyper_grid$n.minobsinnode,
                 verbose = FALSE)
    
    # Make predictions on the validation set
    pred_probs <- predict(model, newdata = x_val, type = "response")
    
    # Compute the AUC for this set of hyperparameters on this fold
    auc <- caret::roc(y_val, pred_probs)$auc
    
    # Add the result to the inner result list
    inner_result_list[[j]] <- list(auc = auc, hyperparameters = hyper_grid)
    
  }
  
  # Find the set of hyperparameters that gave the highest average AUC across the inner folds
  inner_results <- bind_rows(inner_result_list)
  best_hyperparameters <- inner_results %>%
    group_by_all() %>%
    summarize(mean_auc = mean(auc), .groups = "drop") %>%
    arrange(desc(mean_auc)) %>%
    slice(1) %>%
    pull(hyperparameters)
  
  # Train a final gbm model using the best set of hyperparameters and all the training data
  final_model <- gbm(x = x_train, y = y_train,
                     distribution = "bernoulli",
                     interaction.depth
                     