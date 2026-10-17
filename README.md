# Heart Disease Prediction — EDA & Data Preprocessing

## 1. Project Overview

This project focuses on understanding, cleaning, and preparing a Heart Disease dataset for Machine Learning.

The main objective is to perform Exploratory Data Analysis (EDA), identify data quality issues, handle invalid values, visualize feature distributions, analyze relationships between variables, and prepare the dataset for Machine Learning models.

The project covers the following stages:

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Missing and Invalid Value Handling
- Duplicate Detection
- Univariate and Bivariate Analysis
- Outlier Detection
- Feature Engineering
- Categorical Encoding
- Feature Scaling
- Train-Test Split
- Machine Learning Preparation

---

## 2. Dataset Information

The dataset contains information about patients and features associated with heart disease.

### Important Columns

| Column | Description | Data Type |
|---|---|---|
| `Age` | Patient's age | Numerical |
| `Sex` | Patient's sex category | Categorical |
| `ChestPainType` | Type of chest pain | Categorical |
| `RestingBP` | Resting blood pressure | Numerical |
| `Cholesterol` | Serum cholesterol level | Numerical |
| `FastingBS` | Fasting blood sugar indicator | Binary |
| `RestingECG` | Resting electrocardiogram results | Categorical |
| `MaxHR` | Maximum heart rate achieved | Numerical |
| `ExerciseAngina` | Exercise-induced angina indicator | Categorical |
| `Oldpeak` | ST depression measurement | Numerical |
| `ST_Slope` | Slope of the peak exercise ST segment | Categorical |
| `HeartDisease` | Heart disease target indicator | Binary |

Note: The exact meanings and units of measurements should be confirmed from the dataset documentation.

---

## 3. Importing Libraries

The first step is importing the libraries required for data analysis and visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Purpose of Each Library

- **Pandas:** Loads, cleans, and manipulates tabular data.
- **NumPy:** Performs numerical calculations.
- **Matplotlib:** Creates graphs and visualizations.
- **Seaborn:** Creates statistical visualizations such as histograms, boxplots, and heatmaps.
- **Scikit-learn:** Provides preprocessing tools and Machine Learning utilities.

---

## 4. Loading the Dataset

```python
df = pd.read_csv('heart.csv')
```

The `read_csv()` function loads a CSV file into a Pandas DataFrame.

The DataFrame stores the dataset in rows and columns, allowing us to analyze and manipulate the data.

If the filename is different, replace `heart.csv` with the actual filename.

---

## 5. Understanding the Dataset

Before cleaning or transforming the data, it is important to understand its structure.

### View the First Five Rows

```python
df.head()
```

Displays the first five records and helps us understand the columns and their values.

### View the Last Five Rows

```python
df.tail()
```

Displays the last five records.

### Check Dataset Dimensions

```python
df.shape
```

Returns the number of rows and columns.

For example, `(918, 12)` means 918 rows and 12 columns.

### View Column Names

```python
df.columns
```

Displays all column names.

Column names must match exactly when accessing a column. For example, `Age` and `age` are different names in Pandas.

### Check Data Types and Non-Null Counts

```python
df.info()
```

Provides information about column names, data types, non-null values, and memory usage.

### Generate Statistical Summaries

```python
df.describe()
```

Provides numerical statistics, including:

- Count
- Mean
- Standard deviation
- Minimum
- 25th percentile
- Median
- 75th percentile
- Maximum

For categorical columns:

```python
df.describe(include='object')
```

This provides information such as the number of unique categories and their most frequent values.

---

## 6. Checking Missing Values

Missing values occur when a record does not contain a value for a particular feature.

```python
df.isnull().sum()
```

Counts missing values in each column.

To calculate missing-value percentages:

```python
missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print(missing_percentage)
```

### Why Is This Important?

Missing values can affect statistical analysis and may prevent certain Machine Learning algorithms from working correctly.

Possible treatments include:

- Replacing missing numerical values with a mean or median.
- Replacing missing categorical values with a mode or an explicit category.
- Removing rows or columns when justified.

The appropriate treatment depends on the amount of missing data and why the values are missing.

**Important:** A value of zero is not automatically a missing value. It should be treated as invalid only when the dataset's meaning or documentation supports that interpretation.

---

## 7. Detecting Duplicate Records

Duplicate records are rows that repeat the same information.

```python
df.duplicated().sum()
```

Counts duplicate rows.

To remove exact duplicate rows:

```python
df_cleaned = df.drop_duplicates().copy()
```

Verify the result:

```python
df_cleaned.duplicated().sum()
```

### Why Remove Duplicates?

Unintended duplicates can cause repeated observations to influence analysis and model training disproportionately.

However, identical records should be investigated before removal if repeated observations could be legitimate.

---

## 8. Handling Zero Values in RestingBP

In some versions of the Heart Disease dataset, a zero value in `RestingBP` may represent an invalid measurement rather than a genuine blood pressure reading.

First, count the zero values:

```python
(df_cleaned['RestingBP'] == 0).sum()
```

Calculate the mean using only non-zero values:

```python
rbp_mean = df_cleaned.loc[
    df_cleaned['RestingBP'] != 0,
    'RestingBP'
].mean()
```

Replace zeros with the calculated mean:

```python
df_cleaned['RestingBP'] = (
    df_cleaned['RestingBP'].replace(0, rbp_mean)
)
```

Round the result to two decimal places if appropriate:

```python
df_cleaned['RestingBP'] = (
    df_cleaned['RestingBP'].round(2)
)
```

Verify:

```python
(df_cleaned['RestingBP'] == 0).sum()
```

### Understanding the Logic

- `df_cleaned['RestingBP'] != 0` selects non-zero readings.
- `.loc[]` selects the relevant rows and column.
- `.mean()` calculates the average of the selected values.
- `.replace(0, rbp_mean)` replaces zeros with that average.
- `.round(2)` rounds values to two decimal places.

### Why Use the Mean?

The mean provides a simple estimate based on the available non-zero measurements.

However, the mean is sensitive to extreme values. If the data is heavily skewed or contains substantial outliers, the median may be more appropriate.

For a strict Machine Learning workflow, calculate imputation statistics from the training set only.

---

## 9. Handling Zero Values in Cholesterol

The same approach can be applied to `Cholesterol` if the dataset documentation confirms that zero indicates an invalid or unavailable measurement.

Count the zero values:

```python
(df_cleaned['Cholesterol'] == 0).sum()
```

Calculate the mean excluding zeros:

```python
ch_mean = df_cleaned.loc[
    df_cleaned['Cholesterol'] != 0,
    'Cholesterol'
].mean()
```

Replace zero values:

```python
df_cleaned['Cholesterol'] = (
    df_cleaned['Cholesterol'].replace(0, ch_mean)
)
```

Round to two decimal places:

```python
df_cleaned['Cholesterol'] = (
    df_cleaned['Cholesterol'].round(2)
)
```

Verify the result:

```python
(df_cleaned['Cholesterol'] == 0).sum()
```

### Important Consideration

Replacing zeros with the mean is a simple imputation technique, not a guarantee that the original values have been recovered accurately.

In a predictive workflow, imputation should be fitted on training data and then applied to testing data to avoid data leakage.

---

## 10. Separating Numerical and Categorical Features

Numerical and categorical features require different analysis and preprocessing techniques.

```python
numeric_columns = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

categorical_columns = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]
```

The binary columns `FastingBS` and `HeartDisease` should also be examined carefully. `HeartDisease` is the target variable and must not be treated as an input feature.

Check data types:

```python
df_cleaned.dtypes
```

The explicit column lists make the analysis easier to understand and help prevent accidental inclusion of the target in feature transformations.

---

## 11. Univariate Analysis

Univariate analysis studies one variable at a time.

It helps us understand the distribution, range, frequency, and potential unusual values of individual features.

### 11.1 Histogram

```python
sns.histplot(
    df_cleaned['Age'],
    bins=20,
    kde=True
)

plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

### What Is a Histogram?

A histogram divides numerical values into intervals called **bins**.

Each bar represents the number of observations falling within an interval.

- `bins=20` requests 20 intervals.
- `kde=True` adds a smooth estimated density curve.
- `plt.title()` adds a title.
- `plt.xlabel()` labels the horizontal axis.
- `plt.ylabel()` labels the vertical axis.
- `plt.show()` displays the graph.

### 11.2 Plotting Multiple Numerical Features

```python
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(
        df_cleaned[var],
        bins=20,
        kde=True
    )
    plt.title(f'Distribution of {var}')

plt.figure(figsize=(12, 8))

plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)

plt.tight_layout()
plt.show()
```

The `subplot(2, 2, num)` function creates a two-row, two-column arrangement.

The `num` argument specifies the position of each graph.

The function avoids repeating the same plotting code for every feature.

---

## 12. Understanding Bins

Bins control how numerical values are grouped in a histogram.

For example:

```python
sns.histplot(df_cleaned['Age'], bins=5)
```

creates approximately five intervals.

```python
sns.histplot(df_cleaned['Age'], bins=20)
```

creates approximately twenty intervals.

### Why Are Bins Important?

- Fewer bins produce a simpler, less detailed histogram.
- More bins reveal finer patterns but can make the graph noisy.
- The best choice depends on the sample size and distribution.

An automatic option is:

```python
sns.histplot(df_cleaned['Age'], bins='auto')
```

The selected bin count should help reveal the distribution without hiding important patterns or creating unnecessary noise.

---

## 13. Boxplots and Outlier Analysis

A boxplot summarizes the distribution of numerical data and can help identify unusually high or low observations.

```python
sns.boxplot(
    x=df_cleaned['Cholesterol']
)

plt.title('Cholesterol Boxplot')
plt.show()
```

### Understanding the Boxplot

- The line inside the box represents the median.
- The box represents the interquartile range (IQR).
- The whiskers show the range under the plotting rule.
- Points beyond the whiskers are potential outliers.

A potential outlier is not necessarily an error. It may be a genuine measurement.

### IQR Method

```python
Q1 = df_cleaned['Cholesterol'].quantile(0.25)
Q3 = df_cleaned['Cholesterol'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

The IQR is the difference between the 75th and 25th percentiles.

Identify potential outliers:

```python
outliers = df_cleaned[
    (df_cleaned['Cholesterol'] < lower_bound) |
    (df_cleaned['Cholesterol'] > upper_bound)
]

print(outliers)
```

Outliers should be investigated before deciding whether to correct, transform, retain, or remove them.

---

## 14. Categorical Data Analysis

Categorical features represent groups or labels.

### Count Unique Categories

```python
df_cleaned['ChestPainType'].value_counts()
```

This counts observations in each category.

Calculate percentages:

```python
df_cleaned['ChestPainType'].value_counts(
    normalize=True
) * 100
```

### Countplot

```python
sns.countplot(
    x='ChestPainType',
    data=df_cleaned
)

plt.title('Chest Pain Type Distribution')
plt.xticks(rotation=45)
plt.show()
```

A countplot displays the number of observations belonging to each category.

It is useful for identifying common categories and possible class imbalances.

---

## 15. Bivariate Analysis

Bivariate analysis examines the relationship between two variables.

### 15.1 Age vs Maximum Heart Rate

```python
sns.scatterplot(
    x='Age',
    y='MaxHR',
    data=df_cleaned
)

plt.title('Age vs Maximum Heart Rate')
plt.show()
```

A scatterplot helps identify trends, clusters, and unusual observations.

### 15.2 Cholesterol vs Heart Disease

```python
sns.boxplot(
    x='HeartDisease',
    y='Cholesterol',
    data=df_cleaned
)

plt.title('Cholesterol by Heart Disease Status')
plt.show()
```

This compares cholesterol distributions across target classes.

### 15.3 Chest Pain Type vs Heart Disease

```python
sns.countplot(
    x='ChestPainType',
    hue='HeartDisease',
    data=df_cleaned
)

plt.title('Chest Pain Type vs Heart Disease')
plt.xticks(rotation=45)
plt.show()
```

This visualization compares the frequency of heart disease outcomes across chest pain categories.

These plots reveal associations, not necessarily causal relationships.

---

## 16. Correlation Analysis

Correlation measures the strength and direction of a linear relationship between numerical variables.

```python
correlation = df_cleaned[
    numeric_columns
].corr()

print(correlation)
```

### Correlation Heatmap

```python
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.show()
```

### Interpretation

- A value near `+1` indicates a strong positive linear relationship.
- A value near `-1` indicates a strong negative linear relationship.
- A value near `0` indicates little or no linear relationship.

Correlation does not establish causation. It can also miss nonlinear relationships.

The target column `HeartDisease` should be analyzed separately and excluded from the list of input features during model training.

---

## 17. Feature Engineering

Feature engineering creates new features from existing information.

The purpose is to represent useful patterns in a form that a Machine Learning model may be able to learn.

### 17.1 Age Groups

```python
df_cleaned['AgeGroup'] = pd.cut(
    df_cleaned['Age'],
    bins=[0, 30, 45, 60, float('inf')],
    labels=[
        'Young',
        'Adult',
        'MiddleAge',
        'Senior'
    ]
)
```

This converts age values into predefined categories.

The age boundaries are example choices and should be selected according to the project's goals.

### 17.2 BMI Categories

A BMI category can be created when the dataset contains a BMI feature:

```python
df_cleaned['BMICategory'] = pd.cut(
    df_cleaned['BMI'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=[
        'Underweight',
        'Normal',
        'Overweight',
        'Obese'
    ]
)
```

**Note:** The current dataset column list does not include `BMI`. This code applies only if a BMI column is present in another version of the dataset.

### 17.3 Feature Engineering Considerations

New features should have a reasonable interpretation and should be evaluated to determine whether they improve model performance.

Creating additional columns does not automatically improve a model. Unnecessary features can add complexity or introduce redundancy.

---

## 18. Categorical Encoding

Many Machine Learning algorithms require numerical inputs rather than text categories.

One-hot encoding converts each category into a separate binary indicator column.

```python
df_encoded = pd.get_dummies(
    df_cleaned,
    columns=[
        'Sex',
        'ChestPainType',
        'RestingECG',
        'ExerciseAngina',
        'ST_Slope'
    ],
    drop_first=True,
    dtype=int
)
```

### Understanding the Parameters

- `columns` specifies the categorical columns to encode.
- `drop_first=True` drops one category per encoded feature to reduce redundant indicator columns.
- `dtype=int` produces integer-valued indicators.

Inspect the output:

```python
df_encoded.head()
```

```python
df_encoded.columns
```

Binary variables such as `FastingBS` may already be numerical, but their values and meanings should still be checked.

---

## 19. Separating Features and Target

For a Heart Disease classification task:

- **Features (`X`)** contain the input variables.
- **Target (`y`)** contains the value the model is intended to predict.

```python
X = df_encoded.drop(
    columns=['HeartDisease']
)

y = df_encoded['HeartDisease']
```

The target must not be included in `X`. Doing so would create target leakage and invalidate the evaluation.

Check dimensions:

```python
print('Features:', X.shape)
print('Target:', y.shape)
```

---

## 20. Train-Test Split

The dataset should be divided into training and testing subsets before fitting preprocessing steps that learn statistics from data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Parameter Explanation

- `test_size=0.20`: reserves 20% of the data for testing.
- `random_state=42`: makes the split reproducible.
- `stratify=y`: attempts to preserve the target class proportions in both subsets.

The training set is used to fit the model. The test set provides an evaluation on data that was not used for training.

---

## 21. Feature Scaling

Feature scaling adjusts the numerical scales of features.

It is particularly important for algorithms that depend on distances or gradient-based optimization.

### StandardScaler

```python
scaler = StandardScaler()
```

`StandardScaler` standardizes each selected feature using the training data's mean and standard deviation.

The transformed feature is calculated as:

`z = (x - mean) / standard_deviation`

After standardization, the training feature has approximately zero mean and unit standard deviation, provided the feature has non-zero variance.

### Apply Scaling Correctly

```python
scale_columns = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

X_train = X_train.copy()
X_test = X_test.copy()

X_train.loc[:, scale_columns] = (
    scaler.fit_transform(X_train[scale_columns])
)

X_test.loc[:, scale_columns] = (
    scaler.transform(X_test[scale_columns])
)
```

### Why Different Methods?

- `fit_transform()` learns scaling statistics from the training set and transforms it.
- `transform()` applies those same statistics to the test set.

Do not fit the scaler on the test set. Doing so allows information from the test data to influence preprocessing and can lead to data leakage.

Categorical indicator columns are generally left as binary values when using this approach.

---

## 22. Data Leakage Prevention

Data leakage occurs when information that should not be available during training influences the model or its evaluation.

To avoid leakage:

1. Split the data before fitting imputation, scaling, or other learned preprocessing steps.
2. Calculate imputation statistics from training data only.
3. Fit encoders and scalers on training data and reuse them for test data.
4. Never include `HeartDisease` in the input feature matrix.
5. Do not use test results to repeatedly tune the model and then report those same results as an unbiased final evaluation.

For more robust projects, a Scikit-learn `Pipeline` or `ColumnTransformer` can combine preprocessing and modeling in a reproducible workflow.

---

## 23. Final Validation

After preprocessing, check the resulting data:

```python
print('Training features:', X_train.shape)
print('Testing features:', X_test.shape)

print('Training target:', y_train.shape)
print('Testing target:', y_test.shape)
```

Check missing values:

```python
print(X_train.isnull().sum())
print(X_test.isnull().sum())
```

Check target distributions:

```python
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

The feature columns should match between training and testing sets, and the target should remain separate.

---

## 24. Complete Workflow Summary

The overall workflow is:

1. Load the dataset.
2. Inspect rows, columns, data types, and statistical summaries.
3. Identify missing and invalid values.
4. Check and remove unintended duplicates.
5. Investigate zero values in `RestingBP` and `Cholesterol`.
6. Analyze numerical distributions using histograms and KDE.
7. Inspect categorical distributions using countplots.
8. Detect potential outliers using boxplots and the IQR method.
9. Explore feature relationships using scatterplots, boxplots, and heatmaps.
10. Engineer meaningful features where appropriate.
11. Split input features from the target.
12. Split the dataset into training and testing subsets.
13. Fit preprocessing steps on the training data.
14. Encode categorical features and scale suitable numerical features.
15. Validate the final datasets before model training.

---

## 25. Conclusion

Exploratory Data Analysis and preprocessing are essential parts of a Machine Learning project.

EDA helps us understand distributions, patterns, relationships, and unusual observations. Data cleaning addresses quality issues, while feature engineering and encoding prepare useful information for the model. Scaling makes numerical features more comparable for algorithms that benefit from standardized inputs.

The final objective is to produce a clean, well-understood, reproducible dataset that can be used to train and evaluate a Heart Disease classification model.

**Important:** This project is for learning and data analysis. Its results should not be used as a substitute for professional medical diagnosis or treatment.
