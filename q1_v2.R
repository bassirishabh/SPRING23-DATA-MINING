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
load("class_data.RData")


# Make training and test set
y_up=ifelse(y==0,"No","Yes")
y_up = as.factor(y_up)
# x <- as.data.frame(scale(x))
# x <- log(as.data.frame(x))
# process <- preProcess(as.data.frame(x), method=c("range"))
# x <- predict(process, as.data.frame(x))
x=data.frame(x, y_up)

# Logistic regression
set.seed(1)
train=sample(1:nrow(x),250) # by default sample is without replacement
glm.fit=glm(y_up~., data=x,family=binomial, subset=train)
glm.probs=predict(glm.fit,newdata=x[-train,],type="response") 
glm.pred=ifelse(glm.probs >0.5,"Yes","No")
y_up.test=x$y_up[-train]
table(glm.pred,y_up.test)
mean(glm.pred==y_up.test)

## Linear Discriminant Analysis
lda.fit=lda(y_up~.,data=x, subset=train)
lda.fit
x.test=x[-train,]
lda.pred=predict(lda.fit,x.test)
table(lda.pred$class,x.test$y_up)
mean(lda.pred$class==x.test$y_up)

# ## Quadratic Discriminant Analysis
# qda.fit=qda(y_up~.,data=x, subset=train)
# qda.fit
# x.test=x[-train,]
# qda.pred=predict(qda.fit,x.test)
# table(qda.pred$class,x.test$y_up)
# mean(qda.pred$class==x.test$y_up)

## SVM
library(e1071)
?svm
svmfit=svm(y_up~.,data=x,subset=train, kernel="radial", cost=.1)
svm.pred=predict(svmfit, x[-train,])
table(svm.pred, y_up[-train])
mean(svm.pred==y_up[-train])

## RF
# Lets fit a random forest and see how well it performs. 
# We will use the response `medv`, the median housing value (in \$1K dollars)
# rf.boston=randomForest(y_up~.,data=x,subset=train,mtry=500,ntree=500)
# pred=predict(rf.boston,x[-train,])
# some_err <- 1 - mean(pred == x[-train,]$y_up)
# rf.boston
# 
# oob.err=double(60)
# test.err=double(60)
# for(mtry in 1:60){
#   fit=randomForest(y_up~.,data=x,subset=train,mtry=mtry,ntree=500)
#   oob.err[mtry]= fit$err.rate[500]#Mean squared error for 400 trees
#   pred=predict(fit,x[-train,])
#   # print(pred)
#   test.err[mtry] <- 1 - mean(pred == x[-train,]$y_up)
# }
# matplot(1:mtry,cbind(test.err,oob.err),pch=19,col=c("red","blue"),type="b",ylab="Mean Squared Error")
# legend("topright",legend=c("Test", "OOB"),pch=19,col=c("red","blue"))



#With boosting, the number of trees is a tuning parameter, and if we have too many we can overfit. 
#We use cross-validation to choose the number of trees
boost.boston=gbm(y_up~.,data=x[train,],distribution="multinomial",n.trees=500,shrinkage=0.001,interaction.depth=9,cv.fold=10)
n.trees.cv=gbm.perf(boost.boston, method = "cv")
gbmpred=predict.gbm(boost.boston,newdata=x[-train,],n.trees=n.trees.cv, type = "response")
class_names = colnames(gbmpred)[apply(gbmpred, 1, which.max)]
boost_err = 1-mean(class_names == x[-train,]$y_up)
