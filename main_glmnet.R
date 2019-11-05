library(mlflow)
library(glmnet)
set.seed(40)

# Read the train and test file
train <- read.csv("train.csv")
test <- read.csv("test.csv")

train_x <- as.matrix(train[, !(names(train) == "price")])
test_x <- as.matrix(test[, !(names(train) == "price")])
train_y <- train[, "price"]
test_y <- test[, "price"]

alpha <- mlflow_param("alpha", 0.8, "numeric")
lambda <- mlflow_param("lambda", 0.8, "numeric")

with(mlflow_start_run(), {
    model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
    predicted <- predict(model, test_x)

    rmse <- sqrt(mean((predicted - test_y) ^ 2))
    mae <- mean(abs(predicted - test_y))
    r2 <- as.numeric(cor(predicted, test_y) ^ 2)

    message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
    message("  RMSE: ", rmse)
    message("  MAE: ", mae)
    message("  R2: ", r2)

    mlflow_log_param("alpha", alpha)
    mlflow_log_param("lambda", lambda)
    mlflow_log_metric("rmse", rmse)
    mlflow_log_metric("r2", r2)
    mlflow_log_metric("mae", mae)

})
