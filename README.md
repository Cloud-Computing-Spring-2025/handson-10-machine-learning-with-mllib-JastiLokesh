# handson-10-MachineLearning-with-MLlib

#  Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

##  Tasks and Results

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Explanation:**
```python
# Handle missing values in TotalCharges column
df = df.na.fill({"TotalCharges": 0.0})

# Define categorical columns and create StringIndexers
categorical_cols = ["gender", "PhoneService", "InternetService"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"Index", handleInvalid="keep") 
            for col in categorical_cols]

# OneHotEncoder for categorical features
encoder = OneHotEncoder(inputCols=encoder_input_cols, 
                       outputCols=encoder_output_cols,
                       dropLast=True)

# Assemble all features into a single vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
```

**Actual Output:**
```
Processed data with features and ChurnIndex:
+--------------------+----------+
| features           |ChurnIndex|
+--------------------+----------+
|(11,[1,2,3,5,6,8]...| 0.0      |
|(11,[1,2,3,4,7,8]...| 0.0      |
|(11,[1,2,3,5,6,8]...| 1.0      |
|[1.0,59.0,41.5,24...| 0.0      |
|(11,[1,2,3,5,7,9]...| 0.0      |
+--------------------+----------+
only showing top 5 rows
```

---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using accuracy metrics.

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Calculate and display model accuracy.

**Code Explanation:**
```python
# Split data into training (80%) and testing (20%) sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Initialize and train logistic regression model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="ChurnIndex",
    maxIter=10,
    regParam=0.1,
    elasticNetParam=0.0
)
lr_model = lr.fit(train_df)

# Calculate accuracy manually (true positives + true negatives) / total
total = predictions.count()
correct = predictions.filter("ChurnIndex = prediction").count()
accuracy = correct / total
```

**Actual Output:**
```
Training logistic regression model...
Logistic Regression Model Accuracy: 0.73
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Explanation:**
```python
# Initialize Chi-square feature selector for top 5 features
selector = ChiSqSelector(
    numTopFeatures=5,
    featuresCol="features",
    outputCol="selectedFeatures",
    labelCol="ChurnIndex"
)

# Fit the selector to the data and transform
selector_model = selector.fit(df)
selected_df = selector_model.transform(df)
```

**Actual Output:**
```
Selected top 5 features:
+--------------------+----------+
| selectedFeatures   |ChurnIndex|
+--------------------+----------+
|(5,[1,2],[70.0,1.0])| 0.0      |
|(5,[1,2],[31.0,1.0])| 0.0      |
| (5,[1,2],[9.0,1.0])| 1.0      |
|[1.0,59.0,0.0,0.0...| 0.0      |
|(5,[1,3],[54.0,1.0])| 0.0      |
+--------------------+----------+
only showing top 5 rows
```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use hyperparameter tuning to optimize models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Code Explanation:**
```python
# Define models and parameter grids for each model type
models = {
    "LogisticRegression": (
        LogisticRegression(featuresCol="features", labelCol="ChurnIndex"),
        ParamGridBuilder()
            .addGrid(LogisticRegression.regParam, [0.01, 0.1])
            .addGrid(LogisticRegression.maxIter, [10, 20])
            .build()
    ),
    # Similar definitions for other model types...
}

# Cross-validation and evaluation for each model
# Using BinaryClassificationEvaluator with areaUnderROC metric
```

**Actual Output:**
```
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.84
Best Params for LogisticRegression: regParam=0.01, maxIter=20

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.77
Best Params for DecisionTree: maxDepth=10

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.86
Best Params for RandomForest: maxDepth=15, numTrees=50

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.88
Best Params for GBT: maxDepth=10, maxIter=20

Best overall model: GBT with AUC: 0.88
```

---

## Code Explanation

The solution implements a complete machine learning pipeline for predicting customer churn:

1. **Data Preprocessing**
   - Handles missing values in the dataset
   - Converts categorical variables to numerical representations using encoding techniques
   - Creates feature vectors suitable for model training

2. **Model Training**
   - Implements logistic regression as a baseline model
   - Uses appropriate parameters for binary classification
   - Evaluates model performance using accuracy metrics

3. **Feature Selection**
   - Applies Chi-Square test to identify the most predictive features
   - Reduces dimensionality by selecting only the top 5 features
   - Improves model efficiency and interpretability

4. **Model Tuning and Comparison**
   - Tests multiple model types for the classification task
   - Uses grid search to find optimal hyperparameters for each model
   - Compares models using AUC (Area Under ROC Curve) metric
   - Identifies Gradient Boosted Trees as the best performing model with AUC of 0.88

The implementation demonstrates how to properly use PySpark MLlib to build an end-to-end machine learning solution for classification problems with structured data.

---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

```bash
spark-submit churn_prediction.py
```