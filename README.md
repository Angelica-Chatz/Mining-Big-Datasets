## Mining-Big-Datasets
# Assignment using: 
- Windows 10
- R Studio 0.99
- M.R.O 3.2
- Weka 3.8

# Tasks:
- Mashable Online News Popularity dataset
- Use several Classification algorithms to predict the popularity in social networks (popular - non_popular) and compare them using
  'Accuracy' and 'Sensitivity' performance metric
- Ignore the response variable and perform Clustering techniques to determine explanatory and predictive value
- Perform Principal Component Analysis and re-apply the classifying and clustering algorithms in order to observe potential differences   in results


- Steps performed in this project: 

  a) Feature engineering
  
  b) Descriptive Statistics Analysis
  
  c) Missing Values handling
  
  d) Data Standardization
  
  e) Correlation Check 
  
  f) Training - Test split dataset using Bootstrap method (70% - 30%)
  
  g) XGBoost Classifier (Grid Search with 10-fold Cross Validation performed to obtain optimal values)
  
  h) Random Forests Classifier (tune method used to choose optimal No. of trees and No. of variables at each split)
  
  i) Logistic Regression Classifier (along with Stepwise feature selection)
  
  j) SVM Classifier (using Weka default settings, train-test split, 10-fold CV) 
  
  k) Neural Networks (using Weka default settings, train-test split, 10-fold CV) 
  
  l) Confusion Matrix
  
  m) Model Performance Metrics
  
  n) multiple ROC Curve plots 
  
  o) K-Means Clustering (No. of clusters chosen according to Within cluster Sum of Squares plot/'elbow criterion') 
  
  p) EM Clustering (using Weka default settings)
  
  q) Principal Component Analysis 
  
  r) CFsSubsetEval_BestFirst, CFsSubsetEval_GreedyStepwise, OneRAttributeEval (alternative Weka dimension reduction techniques)
  
  s) several visualizations (corrplot, multiple Histograms, XGBoost & Random Forest Variable Importance plot, 
     Principal Components Scree plot, BiPlot PCA, descriptives plots, etc.) 
