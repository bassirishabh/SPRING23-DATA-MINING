# Load required packages
library(xgboost)
library(caret)

load("class_data.Rdata")

y=as.factor(y)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
train <- x[trainIndex,]
test <- x[-trainIndex,]
train_labels <- y[trainIndex]
test_labels <- y[-trainIndex]

# Convert data into DMatrix format

dtrain <- xgb.DMatrix(as.matrix(train), label = train_labels)
dtest <- xgb.DMatrix(as.matrix(test), label = test_labels)

# Set up cross-validation and hyperparameter tuning
xgb_grid <- expand.grid(
  nrounds = c(1000,500),
  max_depth = c(5,9),
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_control <- trainControl(
  method = "cv",
  number = 10,
  search = "grid"
)

# Train the model
set.seed(123)
xgb_model <- train(
  x = train,
  y = train_labels,
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

# Make predictions on the testing data
xgb_pred <- predict(xgb_model, newdata = test)

# Evaluate the model performance
confusionMatrix(xgb_pred, test_labels)