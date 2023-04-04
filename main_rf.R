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

y_up=ifelse(y==0,"No","Yes")
y_up = as.factor(y_up)
x=data.frame(x, y_up)

set.seed(1)
train=sample(1:nrow(x),250) # by default sample is without replacement

## RF
rf.boston=randomForest(y_up~.,data=x,subset=train,mtry=150,ntree=2200)
pred=predict(rf.boston,x[-train,])
accuracy = mean(pred == x[-train,]$y_up)
# rf.boston

