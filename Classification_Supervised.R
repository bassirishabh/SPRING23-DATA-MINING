# Load data
load("~/Desktop/TAMU/Study/Spring23/DataMining/Project/R_Project/class_data.RData")
library(e1071)
# Split data into training and testing sets
set.seed(123)
train_idx <- sample(nrow(x), 0.8*nrow(x))
x_train <- x[train_idx, ]
y_train <- y[train_idx]
x_test <- x[-train_idx, ]
y_test <- y[-train_idx]
# 
# 
x_train <- as.matrix(x_train)
y_train <- as.matrix(y_train)
x_test <- as.matrix(x_test)
y_test <- as.matrix(y_test)
logit_fit <- glm(y_train ~ x_train)
# logit_cv <- glm(x_train, y_train, family = binomial(link = "logit"), type.measure = "class", K = 5)
# logit_best <- logit_fit # In logistic regression, we don't need to tune hyperparameters

logit_pred <- predict(logit_fit, as.data.frame(x_test))

# Convert probabilities to labels
ynew <- as.numeric(logit_pred > 0.5)

# Compute test error
ynew_test <- as.numeric(y_test == 1)
test_error <- mean(ynew != ynew_test)
save(ynew, test_error, file = "~/Desktop/TAMU/Study/Spring23/DataMining/Project/R_Project/output.rdata")
