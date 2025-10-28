# -------------------------------------------------------------
# K-Means Clustering on Supermarket Customer Dataset
# Using pandas + sklearn with data as a list of objects
# -------------------------------------------------------------

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# --- Step 1: Create dataset as list of objects (dictionaries) ---
customers = [
    {"Age": 22, "Income": 15000, "IsStudent": "Yes", "Buys": "No"},
    {"Age": 25, "Income": 29000, "IsStudent": "Yes", "Buys": "No"},
    {"Age": 47, "Income": 48000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 52, "Income": 60000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 46, "Income": 52000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 56, "Income": 72000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 23, "Income": 18000, "IsStudent": "Yes", "Buys": "No"},
    {"Age": 55, "Income": 80000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 60, "Income": 83000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 48, "Income": 62000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 33, "Income": 40000, "IsStudent": "No",  "Buys": "No"},
    {"Age": 43, "Income": 57000, "IsStudent": "No",  "Buys": "Yes"},
    {"Age": 26, "Income": 21000, "IsStudent": "Yes", "Buys": "No"},
    {"Age": 29, "Income": 25000, "IsStudent": "Yes", "Buys": "No"},
    {"Age": 31, "Income": 31000, "IsStudent": "Yes", "Buys": "No"}
]

# --- Step 2: Convert list of dicts into a DataFrame ---
df = pd.DataFrame(customers)
print("Original Dataset:\n", df)

# --- Step 3: Encode categorical features ---
encoder = LabelEncoder()
df['IsStudent'] = encoder.fit_transform(df['IsStudent'])  # Yes=1, No=0
df['Buys'] = encoder.fit_transform(df['Buys'])            # Yes=1, No=0

print("\nEncoded Dataset:\n", df)

# --- Step 4: Select features for clustering ---
# We are clustering customers by Age, Income, and Student status
X = df[['Age', 'Income', 'IsStudent']]

# --- Step 5: Feature scaling (important for fair distance calculation) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 6: Apply K-Means Clustering ---
k = 3  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 7: Show results ---
print("\nCluster Centers (scaled):\n", kmeans.cluster_centers_)
print("\nClustered Data:\n", df)

# --- Step 8: Visualize clusters ---
plt.figure(figsize=(8, 6))
plt.scatter(df['Income'], df['Age'], c=df['Cluster'], cmap='rainbow', s=100)
plt.title('Supermarket Customer Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Age')
plt.grid(True)
plt.show()
