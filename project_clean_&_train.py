import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/Students Social Media Addiction.csv')
df.head(5)

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^0-9a-z_]', '')

print("\nDataset information:")
df.info()

print("\nSummary statistics of numerical variables:")
df.describe()

# Convert categorical variables to appropriate data types if needed
# Check unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

df.isnull().sum()

duplicates = df.duplicated()
print("Number of duplicate rows:", duplicates.sum())
df[duplicates].head()

df.drop_duplicates(inplace=True)
print("Number of duplicates after cleaning:", df.duplicated().sum())

df.drop(columns=['student_id'], inplace=True)
df.drop(columns=['country'], inplace=True)

# Create age groups
bins = [15, 20, 25, 30, 35]
labels = ['16-20', '21-25', '26-30', '31-35']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Categorize social media usage hours
usage_bins = [0, 2, 4, 6, 12, 24]
usage_labels = ['Minimal (0-2h)', 'Moderate (2-4h)', 'High (4-6h)', 'Very High (6-12h)', 'Extreme (12h+)']
df['usage_category'] = pd.cut(df['avg_daily_usage_hours'], bins=usage_bins, labels=usage_labels)

# Categorize sleep hours
sleep_bins = [0, 5, 7, 9, 12]
sleep_labels = ['Poor (<5h)', 'Fair (5-7h)', 'Good (7-9h)', 'Excellent (9h+)']
df['sleep_Category'] = pd.cut(df['sleep_hours_per_night'], bins=sleep_bins, labels=sleep_labels)

top_platforms = df['most_used_platform'].value_counts().nlargest(5).index
df['most_used_platform'] = df['most_used_platform'].apply(lambda x: x if x in top_platforms else 'Other')

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.head()

target = 'addicted_score'
X = df_encoded.drop(columns=[target])
y = df_encoded[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.09, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'social_media_addiction_model.pkl')

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(20, 10))
sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={'alpha':0.6})
plt.xlabel("Actual Addiction Score", fontsize=12)
plt.ylabel("Predicted Addiction Score", fontsize=12)
plt.title("Actual vs Predicted Addiction Scores", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

# residuals = y_test - y_pred
errors = y_test - y_pred

plt.figure(figsize=(20, 10))
sns.histplot(errors, bins=20, kde=True, color='purple')
plt.xlabel("Prediction Error (Actual - Predicted)", fontsize=12)
plt.title("Distribution of Prediction Errors", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)

plt.show()

"""# **Get coefficients and feature names**"""

coefficients = pd.Series(model.coef_, index=X.columns)
coefficients_sorted = coefficients.abs().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=coefficients_sorted[:10], y=coefficients_sorted.index[:10])
plt.title("Top 10 Most Influential Features for Addiction Score")
plt.xlabel("Absolute Coefficient Value (Importance)")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.show()

