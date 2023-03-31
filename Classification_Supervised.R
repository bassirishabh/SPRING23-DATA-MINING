library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

# Load data
library(ISLR)
library(MASS)
load("class_data.RData")

## K-Nearest Neighbors
library(class)
?knn
x=data.frame(x, y)
set.seed(1)
train=sample(1:nrow(x),250) # by default sample is without replacement
# X=Carseats[,c(2,3,4,5,6,8,9)]#extract the continuous covariates
knn.pred=knn(x[train,],x[-train,],y[train],k=50)
table(knn.pred,y[-train])
mean(knn.pred==y[-train])
