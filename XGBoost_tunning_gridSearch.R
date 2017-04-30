OnlineNewsPopularity <- read.csv("C:/Users/Galatation/Desktop/MiningBigDataSets1rstAssignment/OnlineNewsPopularity/OnlineNewsPopularityWeka1.csv")
OnlineNewsPopularity$X <- NULL

Online.1_sampling_vector <- sample(2,nrow(OnlineNewsPopularity),replace=TRUE,prob=c(0.7,0.3))
Online.1_train <- OnlineNewsPopularity[Online.1_sampling_vector==1,]
Online.1_test <- OnlineNewsPopularity[Online.1_sampling_vector==2,]


outcome= Online.1_train[, "shares"]
levels(outcome)
num.class = length(levels(outcome))
levels(outcome) = 1:num.class -1

y = as.matrix(as.integer(outcome)-1)
levels(Online.1_test$shares)<-c(0,1)
Online.1_test$shares<-as.numeric(Online.1_test$shares)-1
Online.1_train$shares<-NULL
train.matrix = as.matrix(Online.1_train)
test.matrix = as.matrix(Online.1_test)


# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(1,0.1, 0.01, 0.001),
  max_depth = c(50, 100, 150,500, 1000),
  gamma = c(1,10,100), 
  colsample_bytree=c(0.1,0.5,1), 
  min_child_weight=c(10,50,100,200,500)
)

library(caret)
# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)


y<- as.factor(y)
y<-make.names(y)

# train the model for each parameter combination in the grid,
#   using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(Online.1_train),y,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree",
  objective = "binary:logistic",
  eval = "auc"
  )

ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) +
  geom_point() +
  theme_bw() +
  scale_size_continuous(guide = "none")

pred <- predict(xgb_train_1, test.matrix)

# confusion matrix
pred <- ifelse(pred == "0",0,1)
pred <- ifelse(pred > 0.5,1,0)
Online.1_test$shares<-as.factor(Online.1_test$shares)
library(caret)
confusionMatrix(Online.1_test$shares,pred)






