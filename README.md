# knn-heart-disease-prediction
 KNN classifier to predict heart disease using Python

## Exploratory Data Analysis
Checked for duplicates, missing values, and implausible zeros (e.g., cholesterol).
Explored the data with univariate and bivariate visualizations to understand feature distributions and relationships with the target variable.

## Data Cleaning & Preprocessing
Treated zero values in cholesterol as missing.
Split the dataset into training and test sets (80/20).

## Built pipelines:
Numeric features: median imputation + MinMax scaling
Categorical features: one-hot encoding

## Feature Engineering & Selection
Created a processed DataFrame with all scaled and encoded features.
Computed a correlation matrix to identify the most predictive variables.

## Modeling
Started with a baseline KNN model (k = 4).
Used grid search with 5-fold cross-validation to tune n_neighbors and the distance metric.

## Evaluation
Evaluated model using accuracy, precision, recall, and F1-score.
Visualized performance with confusion matrix heatmaps.
Final test accuracy: 85.7%, with balanced precision and recall.

## Results
Baseline KNN Test Accuracy: ~77.6%
Tuned KNN Test Accuracy: ~85.7% (best with k = 5, Minkowski distance)
Good generalization with a small train–test gap (less than 1%)
