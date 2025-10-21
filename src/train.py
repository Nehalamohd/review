from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pickle

import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Load the breast cancer dataset
cancer = load_breast_cancer()

# 2. Create a DataFrame for the features (X)
df_features = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

# 3. Create a Series for the target variable (y)
df_target = pd.Series(data=cancer.target, name='target')

# 4. (Optional) Combine features and target into a single DataFrame
#    Map numerical target values to their names for better readability
df_combined = df_features.copy()
df_combined['target'] = df_target
df_combined['target_names'] = df_target.map(lambda x: cancer.target_names[x])

# Display the first few rows of the feature DataFrame
print("Features DataFrame (first 5 rows):")
print(df_features.head())

# Display the first few rows of the target Series
print("\nTarget Series (first 5 rows):")
print(df_target.head())

# Display the first few rows of the combined DataFrame
print("\nCombined DataFrame (first 5 rows):")
print(df_combined.head())
x=df_combined.data
y=df_combined.target
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_sizr=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

with open('model.pkl''wb') as f:
    pickle.dump(model,f)
    
