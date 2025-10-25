# HousePricePrediction
Predict house prices using Python and machine learning. Includes data preprocessing, model training (Linear Regression, Decision Trees, Random Forest), and performance evaluation. Great for learning regression techniques on real-world data.

<details>
## 🧼 Step-by-Step: Clean CSV for ML in Python

### 1. **Load the CSV**
```python
import pandas as pd

df = pd.read_csv('your_file.csv')
```

### 2. **Inspect the Data**
```python
print(df.head())       # Preview
print(df.info())       # Data types and nulls
print(df.describe())   # Stats summary
```

### 3. **Handle Missing Values**
```python
# Drop rows with missing values
df.dropna(inplace=True)

# Or fill missing values
df.fillna(method='ffill', inplace=True)  # forward fill
```

### 4. **Convert Data Types**
```python
df['column_name'] = df['column_name'].astype(float)  # or int, str, etc.
```

### 5. **Encode Categorical Variables**
```python
# One-hot encoding
df = pd.get_dummies(df, columns=['categorical_column'])

# Label encoding (if needed)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['categorical_column'])
```

### 6. **Normalize or Scale Features**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
```

### 7. **Split Features and Target**
```python
X = df.drop('target_column', axis=1)
y = df['target_column']
```

</details>
