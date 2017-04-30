OnlineNewsPopularity <- read.csv("C:/Users/Galatation/Desktop/MiningBigDataSets1rstAssignment/OnlineNewsPopularity/OnlineNewsPopularity.csv")

head(OnlineNewsPopularity)

# Descriptives
str(OnlineNewsPopularity)

Online.1<-OnlineNewsPopularity[-c(1:2, 40:44)]

cols<-c(12:17,30:37)

Online.1[cols]<-lapply(Online.1[cols], factor)

str(Online.1)

sum(is.na(Online.1))

Online.1 <- Online.1[-c(31038),]

library(psych)
describe(Online.1)

options(scipen = 999)


# Multiple HISTS

library(reshape2)
library(ggplot2)
melt.Online.1 <- melt(Online.1)

png(filename="hist1.png", 
    type="cairo",
    units="in", 
    width=15, 
    height=6, 
    pointsize=12, 
    res=150)

ggplot(data = melt.Online.1, aes(x = value)) +
stat_density() +  facet_wrap(~variable, scales = "free")
dev.off()
#dev.copy(png, 'hist.png')


# Check Correlations

library(corrgram)
online.num<-Online.1[-c(12:17,30:37,54)]

png(filename="corrgram.png", 
    type="cairo",
    units="in", 
    width=15, 
    height=6, 
    pointsize=12, 
    res=150)
corrgram(online.num,order=NULL,lower.panel = panel.shade,upper.panel = NULL,text.panel = panel.txt,main="Correlogram")
dev.off()

library(corrplot)
online.num<-Online.1[-c(12:17,30:37,54)]
corrplot.mixed(cor(online.num), lower="circle", 
               upper="color",tl.pos="lt", diag="n", order="hclust", hclust.method="complete")

### VARIABLES NORMALIZATION

# 'H STANDARDIZATION DATA
online.num<-Online.1[-c(12:17,30:37,54)]
online.sc<-as.data.frame(scale(online.num,scale = TRUE, center = TRUE))

describe(online.sc)

# Log numerics

Online.1$n_unique_tokens<-log10(Online.1$n_unique_tokens+1)
Online.1$n_non_stop_unique_tokens<-log10(Online.1$n_non_stop_unique_tokens+1)
Online.1$n_non_stop_words<-log10(Online.1$n_non_stop_words+1)
Online.1$kw_max_min<-log10(Online.1$kw_max_min+1)

describe(Online.1)


# Build Response variable
Online.1[54]<-ifelse(Online.1$shares>1400,1,0)
Online.1[54]<-as.factor(Online.1$shares)
View(Online.1)


#------------------The following lines of code describe Random Forest, Logistic Regression, PCA etc.------------------
#------------------However, we run various Data Ming Techniques (except XGBoost) via Weka.----------------------------

# Set training- test subsets me bootstrap sampling

set.seed(977)
Online.1_sampling_vector <- sample(2,nrow(Online.1),replace=TRUE,prob=c(0.7,0.3))
Online.1_train <- Online.1[Online.1_sampling_vector==1,]
Online.1_test <- Online.1[Online.1_sampling_vector==2,]


# Random Forest

set.seed(2423)
library("randomForest")
library("e1071")
library("caret")
rf_ranges <- list(ntree = c(100, 150), mtry = 2:9)
rf_tune <- tune(randomForest, shares ~ ., data =Online.1_train, ranges = rf_ranges)
rf_tune$best.parameters
rf_best <- rf_tune$best.model
rf_best_predictions <- predict(rf_best, Online.1_test)
varImp(rf_best)
varImpPlot(rf_best,type=2)

mean(rf_best_predictions == Online.1_test[,54])

(confusion_matrix <- table(predicted = rf_best_predictions, actual = Online.1_test[,54]))

Accuracy<-(confusion_matrix[1,1]+confusion_matrix[2,2])/sum(confusion_matrix)
paste('Accuracy',Accuracy)


# ROC evaluation metric

library(pROC)
rf_best_predictions2<-as.numeric(rf_best_predictions)

roc1 <- roc(Online.1_test$shares,rf_best_predictions2)
(auc <- roc1$auc)

plot(roc1,col="purple",main="Random Forests ROC",legacy.axes=TRUE)


## LOGISTIC REGRESSION

set.seed(2423)

full<-glm(shares~.,family = binomial,data = Online.1_train) 

null<-glm(shares~1,family=binomial,data = Online.1_train)

step(null, scope=list(lower=null, upper=full), direction="forward")
step(full, data=Online.1_train, direction="backward")
step(null, scope = list(upper=full), data=Online.1_train, direction="both")

best_fit1 <- glm(formula = shares ~ data_channel_is_world + is_weekend + data_channel_is_entertainment + 
                   kw_avg_avg + kw_max_avg + data_channel_is_socmed + self_reference_min_shares + 
                   weekday_is_wednesday + rate_positive_words + kw_min_min + 
                   data_channel_is_tech + weekday_is_friday + kw_max_min + n_tokens_title + 
                   num_keywords + num_self_hrefs + min_negative_polarity + title_sentiment_polarity + 
                   abs_title_subjectivity + title_subjectivity + data_channel_is_lifestyle + 
                   n_non_stop_unique_tokens + global_rate_negative_words + self_reference_avg_sharess, 
                 family = binomial, data = Online.1_train) 

summary(best_fit1)  



fitted.results <- predict(best_fit1,newdata=Online.1_test,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

(confusion_matrix <- table(predicted=fitted.results,observed=Online.1_test$shares))

Accuracy<-(confusion_matrix[2,2]+confusion_matrix[1,1])/sum(confusion_matrix)
paste('Accuracy',Accuracy)

# ROC curve kai AUC
library(pROC)

roc2 <- roc(Online.1_test$shares,fitted.results)
(auc <- roc1$auc)

plot(roc2,col="red",main="Online.1_Test ROC",legacy.axes=TRUE)


# COMPARING RF-LOG-REG CLASSIFIERS

par(mar=c(0, 0, 0, 0))
plot(roc1,col="orange",main="Random Forests ROC",legacy.axes=TRUE)
plot(roc2,col="purple",main="LOG REG vs Random Forests ROC",xaxt = "n", yaxt = "n",add=TRUE)
legend("center", ncol=5, c("RForest","SVM"), col = c("orange","purple"), bty = "n")


### CLUSTERING ###

#dataset to use for clustering
dataset <- read.csv("C:/Users/AGGELIKI/Desktop/OnlineNewsPopularity/OnlineNewsPopularity.csv")

#scale data
str(dataset)

dataset <- dataset[,3:61]

cols<-c(12:17,30:37)

dataset[cols]<-lapply(dataset[cols], factor)

dataset[59]<-ifelse(dataset$shares>1400,1,0)
dataset[59]<-as.factor(dataset$shares)

data_num <- dataset[,-c(12:17,30:37,59)]

data_sc <- scale(data_num,scale = TRUE, center = TRUE) # standardize variables


#Method 1 

# k means algorithm
library(NbClust)

nc<-NbClust(data_sc,min.nc=2,max.nc=6, method="kmeans")
par(mfrow = c(1, 1))
barplot(table(nc$Best.n[1,]),xlab="Number of Clusters", ylab="Number of Criteria",
        main="Number of Clusters Chosen")


#predict number of clusters
wssplot<-function(data,nc=8,seed=1234){
  wss<-(nrow(data)-1)*sum(apply(data,2,var))
  for(i in 2:nc){
    set.seed(seed)
    wss[i]<-sum(kmeans(data,centers=i)$withinss)
  }
  plot(1:nc,wss,type="b",xlab="NumberofClusters",
       ylab="Withingroupssumofsquares")
}

wssplot(data_sc)


#suggests a 2 cluster solution

set.seed(333)
fit.kmeans<-kmeans(data_sc,2,nstart=50)
fit.kmeans
(cm_kmeans<-  table(type=dataset$shares,cluster=fit.kmeans$cluster))
accuracy <- round(((cm_kmeans[1,2]+cm_kmeans[2,1])/(cm_kmeans[1,2]+cm_kmeans[2,1]+cm_kmeans[1,1]+
                                                      cm_kmeans[2,2])),2)
paste("Accuracy:",accuracy)

#determine variable means for each cluster in the original metric.

aggregate(dataset[-c(1,2)], by=list(cluster=fit.kmeans$cluster), mean)


### PCA ###

# create PCs 
Online.pca <- read.csv("C:/Users/AGGELIKI/Desktop/OnlineNewsPopularity/OnlineNewsPopularity.csv")

PCs_out <- prcomp(Online.pca[-c(1,2)], scale = TRUE)

# Choose PCs
summary(PCs_out)

# PCs plot
library(factoextra)

fviz_screeplot(PCs_out, ncp=10, choice="eigenvalue")

fviz_pca_var(PCs_out, col.var="contrib") +
  scale_color_gradient2(low="white", mid="blue", 
                        high="red", midpoint=50) + theme_minimal()

# new data with PC1, PC2, PC3, type
new_data_sample <- data.frame(PCs_out$x[,c(1:3)])
new_data <- cbind(Online.pca$shares,new_data_sample)
colnames(new_data)[1] <- "type"