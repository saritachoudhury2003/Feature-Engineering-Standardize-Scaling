📊 Standardized Scaling using StandardScaler
📌 Project Overview
This project demonstrates how to apply Standardization (Z-score Normalization) using Scikit-learn's `StandardScaler` on the Social Network Ads dataset.
The objective is to transform numerical features so they have:
Mean = 0
Standard Deviation = 1
Feature scaling is an essential preprocessing step for machine learning algorithms such as Logistic Regression, KNN, SVM, and Neural Networks.
---
📂 Dataset
Dataset: Social Network Ads
Features
Age
EstimatedSalary
Target Variable
Purchased (0 = No, 1 = Yes)
---
🚀 Libraries Used
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
---
📖 Workflow
1. Load the Dataset
```python
df = pd.read_csv("Social_Network_Ads.csv")
```
2. Separate Features and Target
```python
X = df.drop("Purchased", axis=1)
y = df["Purchased"]
```
3. Split the Dataset
```python
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
```
70% Training Data
30% Testing Data
`random_state=0` ensures the same train-test split every time the code is executed.
4. Apply StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```
`fit_transform()` learns the mean and standard deviation from the training data and scales it.
`transform()` scales the test data using the same values learned from the training data.
5. Convert Back to DataFrame
```python
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)
```
This preserves the original column names.
6. Verify Scaling
```python
np.round(x_train_scaled.describe(), 1)
```
The `1` rounds the output to one decimal place.
7. Visualize the Distribution
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

ax1.set_title("Age Distribution Before Scaling")
sns.kdeplot(x_train["Age"], ax=ax1)

ax2.set_title("Age Distribution After Standard Scaling")
sns.kdeplot(x_train_scaled["Age"], ax=ax2)

plt.show()
```
---
📈 Results
Before scaling:
Features had different ranges.
After scaling:
Mean ≈ 0
Standard Deviation ≈ 1
Distribution shape remains the same.
Only the scale changes.
---
📚 Standardization Formula
[
z = \frac{x-\mu}{\sigma}
]
Where:
x = Original value
μ = Mean
σ = Standard deviation
---
✅ Key Learnings
Applied StandardScaler correctly.
Understood the difference between `fit_transform()` and `transform()`.
Preserved column names after scaling.
Verified scaling using descriptive statistics.
Compared distributions before and after scaling.
---
🛠 Technologies
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
