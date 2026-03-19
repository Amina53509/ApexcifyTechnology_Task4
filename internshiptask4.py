# TASK 4: Basic Descriptive Statistics + Visualizations
# -----------------------------------------------------
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset (Iris dataset for demo, can replace with 'tips' or 'titanic')
df = sns.load_dataset("iris")

# Step 2: Preview dataset
print("Dataset Preview:")
print(df.head(), "\n")

# Step 3: Descriptive statistics
print("Descriptive Statistics (describe):")
print(df.describe(), "\n")

# Step 4: Custom statistics
print("Mean:\n", df.mean(numeric_only=True), "\n")
print("Median:\n", df.median(numeric_only=True), "\n")
print("Minimum:\n", df.min(numeric_only=True), "\n")
print("Maximum:\n", df.max(numeric_only=True), "\n")
print("Standard Deviation:\n", df.std(numeric_only=True), "\n")

# Step 5: Bonus - Missing values check
print("Missing Values Check:")
print(df.isnull().sum())

# Step 6: Visualizations
# Histogram for each numeric column
df.hist(figsize=(10, 8), bins=15, edgecolor='black')
plt.suptitle("Histograms of Iris Dataset Features", fontsize=14)
plt.show()

# Boxplots to show spread and outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(columns=['species']))
plt.title("Boxplots of Iris Dataset Features")
plt.show()

# Pairplot to show relationships between features
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()