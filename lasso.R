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
train=sample(1:nrow(x),250)

x_train = x[train,]
x_test = x[-train,]
y_train=y[train]
y_test=y[-train]

x_train=scale(x_train)

y_train_up=ifelse(y_train==0,"No","Yes")
y_train_up = as.factor(y_train_up)

y_test_up=ifelse(y_test==0,"No","Yes")
y_test_up = as.factor(y_test_up)

# Lasso Regression
fit.lasso=glmnet(x_train,y_train, alpha = 1, lambda = 0.009)
# plot(fit.lasso,xvar="lambda",label=TRUE)
# cv.lasso=cv.glmnet(x_train,y_train)
# plot(cv.lasso)
# lasso.best.lambda=cv.lasso$lambda.min # find the best lambda value corresponding to min cv.error
# log(lasso.best.lambda)
# min(cv.lasso$cvm) # min cv.error
# sx = coef(fit.lasso)
outMat = fit.lasso$beta
# tabulate(fit.lasso$beta@i + 1)
entries <- rownames(outMat)[rowSums(outMat != 0) == 1]

# x_train = x_train[, entries]

# Adding y values to x
x_train=data.frame(x_train, y_train_up)
x_test=data.frame(x_test, y_test_up)

## RF
# Lets fit a random forest and see how well it performs. 
# We will use the response `medv`, the median housing value (in \$1K dollars)
# rf_train = sample(1:nrow(x_train),250)
# rf.boston=randomForest(y_train_up~.,data=x_train, subset = rf_train,mtry=22,ntree=2000)
# pred=predict(rf.boston,x_test)
# some_err <- 1 - mean(pred == x_test$y_test_up)
# rf.boston

#With boosting, the number of trees is a tuning parameter, and if we have too many we can overfit. 
#We use cross-validation to choose the number of trees
boost.boston=gbm(y_train_up~.,data=x_train,distribution="multinomial",n.trees=500,shrinkage=0.0001,interaction.depth=20,cv.fold=10)
n.trees.cv=gbm.perf(boost.boston, method = "cv")
gbmpred=predict.gbm(boost.boston,newdata=x_test,n.trees=n.trees.cv, type = "response")
class_names = colnames(gbmpred)[apply(gbmpred, 1, which.max)]
boost_err = 1-mean(class_names == x_test$y_test_up)

# # find the best lambda value corresponding to 1 standard error above the minimum MSE
# # usually more conservative (fewer variables) solution than the minimum MSE
# lasso.lambda.1se=cv.lasso$lambda.1se 
# 
# #find coefficients of best model
# best_model <- glmnet(x_train, y_train, alpha = 1, lambda = 0.00091188)
# coef(best_model)
