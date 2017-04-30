
#### XGBOOST ####
set.seed(123)
OnlineNewsPopularityWeka1 <- read.csv("C:/Users/Galatation/Desktop/MiningBigDataSets1rstAssignment/OnlineNewsPopularity/OnlineNewsPopularityWeka1.csv")

OnlineNewsPopularityWeka1$X<-NULL

levels(OnlineNewsPopularityWeka1$shares)[levels(OnlineNewsPopularityWeka1$shares)=="popular"] <- "1"
levels(OnlineNewsPopularityWeka1$shares)[levels(OnlineNewsPopularityWeka1$shares)=="non popular"] <- "0"

Online.1_sampling_vector <- sample(2,nrow(OnlineNewsPopularityWeka1),replace=TRUE,prob=c(0.8,0.2))

Online.1_train <- OnlineNewsPopularityWeka1[Online.1_sampling_vector==1,]
Online.1_test <- OnlineNewsPopularityWeka1[Online.1_sampling_vector==2,]



label<-as.numeric(Online.1_train[[54]])-1

Online.1_train$shares<-NULL


train.matrix = as.matrix(Online.1_train)

test.matrix = as.matrix(Online.1_test[-c(54)])

param <- list("objective" = "binary:logistic",
                         "eval_metric" = "auc",
                             "max_depth" = 50,
                             "eta" =0.01,
                             "gamma" = 0.01,                
                             "subsample" = 0.5,
                             "colsample_bytree" =0.5,
                             "min_child_weight"=100,"max_delta_step"=100,"lambda"=3
              )

library("xgboost", lib.loc="~/R/win-library/3.2")

bst.cv <- xgboost(param=param, data=train.matrix, label=label,nrounds=5000,verbose=TRUE)


pred <- predict(bst.cv, test.matrix)

# confusion matrix
pred <- ifelse(pred > 0.5,1,0)
Online.1_test$shares<-as.factor(Online.1_test$shares)

library(caret)
confusionMatrix(Online.1_test$shares,pred)

##########################3
(confusion_matrix <- table(predicted=pred,observed=Online.1_test$shares))
Accuracy<-(confusion_matrix[2,2]+confusion_matrix[1,1])/sum(confusion_matrix)
paste('Accuracy',Accuracy)
######################################

model = xgb.dump(bst.cv, with.stats=TRUE)
# get the feature real names
feature_names = dimnames(train.matrix)[[2]]
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = feature_names, model=bst.cv)
head(importance_matrix)

# #improvement in the interability of feature importance data.table
# importance_Raw <- xgb.importance(feature_names = feature_names, model = bst.cv, data=train.matrix, label=label)
# #cleaning for better display
# importanceClean <- importance_Raw[,':='(Cover=NULL, Frequency=NULL)]
# head(importanceClean)

# plotting the feature importance
gp = xgb.plot.importance(importance_matrix, numberOfClusters = 2)
print(gp) 
#Features are divided in 2 clusters: the interesting features and the others.
#Feature Importance plot is useful to select only best features with highest correlation to the outcome(s).
#To improve model fitting performance (time or overfitting), less important features can be removed.


#additional plot
#xgb.plot.importance(importance_matrix)

    