library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

# Load data
library(ISLR)
library(MASS)
library(tree)
library(randomForest)
library(gbm)
library(glmnet)
library(coefplot)
load("class_data.RData")


set.seed(1)
train=sample(1:nrow(x),250) # by default sample is without replacement

x_train = x[train,]
x_test = x[-train,]
y_train=y[train]
y_test=y[-train]

x_train=scale(x_train)

# Lasso Regression
fit.lasso=glmnet(x_train,y_train, alpha = 1, lambda = 0.0009)
# # plot(fit.lasso,xvar="lambda",label=TRUE)
# # cv.lasso=cv.glmnet(x_train,y_train)
# # plot(cv.lasso)
# # lasso.best.lambda=cv.lasso$lambda.min # find the best lambda value corresponding to min cv.error
# # log(lasso.best.lambda)
# # min(cv.lasso$cvm) # min cv.error
# # sx = coef(fit.lasso)
outMat = fit.lasso$beta
# # tabulate(fit.lasso$beta@i + 1)
entries <- rownames(outMat)[rowSums(outMat != 0) == 1]

# x_train = x_train[, entries]

# Adding y values to x
x_train=data.frame(x_train, y_train)
x_test=data.frame(x_test, y_test)

# Fit the gbm model
boost.boston=gbm(y_train~.,data=x_train,distribution="bernoulli",n.trees=500,
                 shrinkage=0.0001,interaction.depth=9, bag.fraction = 0.5, cv.folds = 10, verbose = T)
n.trees.cv=gbm.perf(boost.boston, method = "cv")
gbmpred=predict.gbm(boost.boston,newdata=x_test,n.trees=n.trees.cv, type = "response")
gbmpred[gbmpred < 0.5] = 0
gbmpred[gbmpred >= 0.5] = 1
matrix.train <- table(gbmpred, x_test$y_test)
classification_error = (matrix.train[1, 2] + matrix.train[2, 1])/nrow(x_test)
classification_accuracy = mean(gbmpred == x_test$y_test)
importance <- summary(boost.boston, n.tree = n.trees.cv, main = "RELATIVE INFLUENCE OF ALL PREDICTORS")

# Subset the data frame to include only variables with rel.inf > 0.1
important_vars <- importance[importance$rel.inf > 0.3, ]

# Extract the variable names as a character vector
selected_features <- as.character(important_vars$var)

# Train a Lasso model on the training data using the selected subset of features
x_train <- x_train[,selected_features]
y_train <- y_train
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5)

# Use the trained Lasso model to predict on the test data and evaluate its performance
x_test <- test[,selected_features]
y_test <- test$target
lasso_pred <- predict(lasso_model, newx = x_test, s = "lambda.min")
lasso_pred_class <- ifelse(lasso_pred > 0, 1, 0)
confusionMatrix(table(lasso_pred_class, y_test))

